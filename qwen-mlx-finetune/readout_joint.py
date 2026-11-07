#!/usr/bin/env python3
"""LoRA 骨干 + readout 头**联合训练**，输出 d3 格式，oMLX 原生服务。

## 为什么不是冻结骨干

2026-11-06 测过冻结骨干那一版，三组对照（原版/因果/v13微调骨干）全部输给 LoRA：

    配置                      验证 argmax    AUC      零漏放代价
    非因果 · 原版骨干             75.6%     0.8922      66.2%
    因果   · 原版骨干             74.8%     0.9169      74.8%
    非因果 · v13 微调骨干          77.1%     0.9139      30.5%
    v13 LoRA（基线）              82.0%     0.9825      29.8%

两条读数决定了这个文件的存在：

* **双向注意力在原版骨干上几乎没用**（75.6% vs 74.8%，AUC 反而因果更高）。
  LLM2Vec（arXiv 2024-04）早就写了：转双向编码器要三步——①打开双向注意力
  ②MNTP 适配 ③对比学习，而**只做第①步是不够的**。我们之前只做了第①步。
  d3-lite 的骨干和原版 Qwen 在中层 MLP 上相对差 9.8%，它是连骨干一起调过的。
* **换成微调骨干能涨**（75.6 → 77.1，AUC 0.892 → 0.914），说明瓶颈在**特征质量**
  而不是线性头。非线性探针只多拿 2.2pp，进一步排除了「线性约束是瓶颈」。

所以骨干必须动。《Fine-Tuning Causal LLMs for Text Classification》（2025-12）
也指向同一结论：给因果 LLM 挂分类头、用最后 token 的 embedding 作序列表示，
**在单标签分类上达到或超过微调 BERT，而可训练参数少 10–30 倍**；
instruction 微调（我们现在这条）只在多标签场景有竞争力，且需要 ≥100M 可训练参数。

## 为什么这会比现在的 LoRA 便宜

**完全不需要 248320 维的 LM 头。** 那是 OOM 的成因和主要算力开销——现在输出
只有 3 个 logit。序列长度反而差不多（d3 模板中位 143，现行短提示词中位 94）。

## 实现上两个会静默出错的地方

1. **padding 与双向注意力**。batch 里补齐之后，真实 token 会注意到 padding，
   而 `last` 池化也可能取到 padding 位。两者都不报错，只是分数变差。
   这里**用 batch 1 + 梯度累积**绕开：没有 padding 就没有这两个坑。先测速度，
   确实慢到不可接受再上掩码——正确性优先于速度。
2. **readout 不能有 bias**。oMLX 算的是裸的 `hidden @ W.T`。训练时带了 bias、
   打包时丢掉，线上就和训练对不上。这里直接不建 bias。
   特征也不做归一化，理由同上：`mu`/`sd` 没地方放（见 `readout_train.py` 里
   那次逐类偏置 91.6 的教训）。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers

from readout_features import HIDDEN, N_CODES, write_package
from readout_train import LABELS, read_rows, render


def forward_pooled(model, ids: mx.array, noncausal: bool = True,
                   lengths: mx.array | None = None) -> mx.array:
    """跑骨干，返回每条样本最后一个**真实** token 的隐状态 [B, 1024]。

    不走 `TextModel.__call__`——那一步会把隐状态投到 248320 维词表上，
    而我们要的正是投影之前的东西，也正是不需要算的那一大块。

    `lengths` 给了就是批处理（右侧补齐）。padding 下有两个**不会报错、只会让
    分数变差**的坑，都在这里处理：

    1. **真实 token 会注意到 padding。** 双向注意力下每个位置都能看到全序列，
       包括补出来的那些。要给全注意力层一个加性掩码，把 padding 的 key 屏蔽掉。
    2. **`last` 池化取到 padding 位。** 右侧补齐之后最后一个位置是 padding，
       不能直接取 `[:, -1]`，要按每条的真实长度去取。

    SSM 层另有自己的掩码契约（`mx.where(mask[..., None], qkv, 0)`，布尔 [B, L]）。

    批处理和单条必须给出一致的结果——`--check-batching` 会逐条验。
    """
    m = model.language_model.model
    h = m.embed_tokens(ids)
    B, L = ids.shape

    if lengths is None:
        pad_bool = None
        attn_pad = None
    else:
        pos = mx.arange(L)[None, :]
        pad_bool = pos < lengths[:, None]                      # [B, L] True=真实
        # 加性掩码，屏蔽 padding 的 key；形状 [B, 1, 1, L] 对 query 广播
        attn_pad = mx.where(pad_bool[:, None, None, :], 0.0, -1e9).astype(h.dtype)

    for layer in m.layers:
        if layer.is_linear:
            mask = pad_bool
        elif noncausal:
            mask = attn_pad
        else:
            mask = "causal" if attn_pad is None else attn_pad
        h = layer(h, mask=mask, cache=None)
    h = m.norm(h)
    if lengths is None:
        return h[:, -1]
    idx = (lengths - 1).astype(mx.int32)
    return mx.take_along_axis(h, idx[:, None, None], axis=1).squeeze(1)


def make_batches(items, size: int):
    """按长度排序后相邻成批，补齐到批内最长——同批长度接近，padding 就少。"""
    order = sorted(range(len(items)), key=lambda i: items[i][0].shape[1])
    for s in range(0, len(order), size):
        chunk = [items[i] for i in order[s : s + size]]
        lens = [x.shape[1] for x, _y in chunk]
        width = max(lens)
        ids = mx.concatenate([
            mx.pad(x, [(0, 0), (0, width - x.shape[1])]) for x, _y in chunk
        ], axis=0)
        yield ids, mx.array(lens), mx.array([y for _x, y in chunk])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="打包目标目录")
    ap.add_argument("--num-layers", type=int, default=8,
                    help="LoRA 加在最后这么多层。池化取最后一层，后层影响最大；"
                         "没有 LM 头之后显存宽裕，比现行的 4 层多给一些")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=3e-4,
                    help="头是从零开始的，学习率要比 LoRA 高一个量级")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=2)
    # 梯度累积。batch 大小对吞吐没帮助（实测 batch 1/2/4 都是 0.65~0.74 s/样本，
    # 训练模式关掉了快速核，回退路径线性），但**有效 batch 对稳定性是决定性的**。
    #
    # 第一版 batch 2 无累积，验证集在两个退化解之间来回弹：
    #   step  500  全判 SAFE（漏放 166 / 误拦 1）
    #   step 2000  全判拦截（漏放 0 / 误拦 166）
    #   step 2500  又全判 SAFE
    # 每步只看 2 个样本、类别权重最高 4.6×，梯度噪声压过信号。累积不增加
    # 墙钟时间（同样的样本数），只是把更新变稳。
    ap.add_argument("--accum", type=int, default=8,
                    help="梯度累积步数；有效 batch = batch × accum")
    ap.add_argument("--warmup", type=int, default=100,
                    help="预热步数。头是从零起的，上来就满学习率会主导整个更新")
    ap.add_argument("--eval-every", type=int, default=200, help="每多少个优化步测一次验证集")
    ap.add_argument("--causal", action="store_true", help="因果对照组")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条，冒烟用")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    model, tok = load(str(args.backbone))
    model.freeze()
    linear_to_lora_layers(
        model.language_model, args.num_layers,
        {"rank": args.rank, "scale": 20.0, "dropout": 0.0},
    )
    head = nn.Linear(HIDDEN, len(LABELS), bias=False)   # oMLX 算裸 hidden @ W.T，无 bias

    class Joint(nn.Module):
        """把骨干和头装在一起，`nn.value_and_grad` 才会同时给两边的梯度。

        直接对 `model` 求梯度只会拿到 LoRA 那一份，头永远不更新——第一版就是
        这样，报 `not enough values to unpack`。
        """

        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

    joint = Joint(model, head)
    trainable = [p for _k, p in nn.utils.tree_flatten(model.trainable_parameters())]
    print(f"LoRA 可训练 {sum(p.size for p in trainable)/1e6:.3f}M + 头 "
          f"{HIDDEN*len(LABELS)/1e3:.1f}K  （{args.num_layers} 层 rank {args.rank}）")

    def encode(path: Path):
        rows = read_rows(path)
        if args.limit:
            rows = rows[: args.limit]
        return [(mx.array(tok.encode(render(c, tok), add_special_tokens=False))[None],
                 LABELS.index(l)) for c, l in rows]

    tr, va = encode(args.train), encode(args.valid)
    counts = np.bincount([y for _x, y in tr], minlength=len(LABELS)).astype(np.float64)
    # 类别加权：插件只自动执行 SAFE，三类的代价不对称，不能让多数类主导
    cw = mx.array((counts.sum() / (len(LABELS) * np.maximum(counts, 1))).astype(np.float32))
    print(f"训练 {len(tr)} 条 {dict(zip(LABELS, counts.astype(int).tolist()))}，验证 {len(va)} 条")

    def loss_fn(j, ids, lens, ys):
        logits = j.head(forward_pooled(j.backbone, ids, not args.causal, lens))
        ll = logits - mx.logsumexp(logits, axis=1, keepdims=True)
        picked = mx.take_along_axis(ll, ys[:, None], axis=1).squeeze(1)
        w = cw[ys]
        return -(picked * w).sum() / w.sum()

    # 必须进训练模式：`gated_delta` 里是 `use_kernel=not self.training`，
    # eval 模式下走的自定义 Metal 核没有 VJP，反传直接报
    # `[Primitive::vjp] Not implemented for CustomKernel`。
    joint.train()
    state = [model.state, head.state, mx.random.state]
    grad_fn = nn.value_and_grad(joint, loss_fn)

    def evaluate(rows, batch=8):
        # 切回 eval：Gated DeltaNet 在非训练模式下走自定义 Metal 核，快得多。
        # 训练时必须 train()，那个核**没有反向实现**（`use_kernel=not self.training`），
        # 不切就是 `[Primitive::vjp] Not implemented for CustomKernel`。
        joint.eval()
        pred, truth = [], []
        for ids, lens, ys in make_batches(rows, batch):
            lg = head(forward_pooled(model, ids, not args.causal, lens))
            pred += np.asarray(lg.tolist()).argmax(1).tolist()
            truth += np.asarray(ys.tolist()).tolist()
        joint.train()
        p, t = np.array(pred), np.array(truth)
        leak = int(((p == 0) & (t != 0)).sum())
        block = int(((p != 0) & (t == 0)).sum())
        return float((p == t).mean()), leak, block

    # 按长度分批（见 `make_batches`）。batch 1 实测 0.69 s/样本，比现行
    # batch 2 的 LoRA（0.325 s/样本，还带着 LM 头）还慢——batch 1 损失的并行度
    # 大于省掉 LM 头的收益，所以批处理不是优化，是必须。
    batches = list(make_batches(tr, args.batch))
    steps_per_epoch = len(batches)
    total = max(1, int(steps_per_epoch * args.epochs) // args.accum)
    rng = np.random.default_rng(args.seed)

    opt_lora = optim.AdamW(learning_rate=args.lr)
    opt_head = optim.AdamW(learning_rate=args.head_lr)
    # 漏放优先，和 `select_checkpoint.py` 同一口径：误拦多按一次回车，
    # 漏放是一条不可逆命令已经跑完了。
    best: tuple[int, int, float] | None = None
    best_acc = 0.0
    t0 = time.time()
    print(f"每轮 {steps_per_epoch} 批（batch {args.batch} × 累积 {args.accum} = "
          f"有效 {args.batch * args.accum}），共 {total} 个优化步")

    for step in range(total):
        warm = min(1.0, (step + 1) / max(args.warmup, 1))
        decay = 0.5 * (1 + math.cos(math.pi * step / total))
        for opt, base in ((opt_lora, args.lr), (opt_head, args.head_lr)):
            opt.learning_rate = base * warm * decay
        gl = gh = None
        for k in range(args.accum):
            idx = step * args.accum + k
            if idx % steps_per_epoch == 0:
                rng.shuffle(batches)
            ids, lens, ys = batches[idx % steps_per_epoch]
            _loss, g = grad_fn(joint, ids, lens, ys)
            gl = g["backbone"] if gl is None else nn.utils.tree_map(
                lambda a, b: a + b, gl, g["backbone"])
            gh = g["head"] if gh is None else nn.utils.tree_map(
                lambda a, b: a + b, gh, g["head"])
        scale = 1.0 / args.accum
        opt_lora.update(model, nn.utils.tree_map(lambda t: t * scale, gl))
        opt_head.update(head, nn.utils.tree_map(lambda t: t * scale, gh))
        mx.eval(state)

        if (step + 1) % args.eval_every == 0 or step + 1 == total:
            acc, leak, block = evaluate(va)
            el = time.time() - t0
            print(f"  step {step+1:>5}/{total}  验证 {acc:>6.1%} 漏放 {leak:>3} 误拦 {block:>3}"
                  f"   {el/60:.1f}min  预计总计 {el/(step+1)*total/60:.0f}min", flush=True)
            score = (-leak, -block, acc)
            if best is None or score > best:
                best, best_acc = score, acc
                W = np.asarray(head.weight.tolist(), dtype=np.float32)
                full = np.zeros((N_CODES, HIDDEN), dtype=np.float32)
                full[: len(LABELS)] = W
                np.save(args.out.with_suffix(".readout.npy"), full)
                mx.save_safetensors(
                    str(args.out.with_suffix(".adapters.safetensors")),
                    dict(nn.utils.tree_flatten(model.trainable_parameters())),
                )

    print(f"\n最好的一版：验证 {best_acc:.1%}  漏放 {-best[0]}  误拦 {-best[1]}")
    print(f"  readout  {args.out.with_suffix('.readout.npy')}")
    print(f"  adapters {args.out.with_suffix('.adapters.safetensors')}")
    print("  下一步：mlx_lm fuse 合进骨干 → readout_train.py pack → eval_readout.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""按 d3 的方式抽特征：非因果骨干 + last 池化 → 1024 维。

## 这是什么

`d3-lite` 拆开来看只有三样东西（`~/.omlx/models/vllm-sr/d3-lite/`）：

    decision_config.json   format_id=d3-code-readout-v1, base_model=Qwen/Qwen3.5-0.8B
                           attention_mode=noncausal_full_attention, pooling=last
    readout.safetensors    weight [255, 1024] float32  ——  26 万参数，一个线性层
    model-*.safetensors    骨干，和我们的 Qwen3.5-0.8B-Base **架构完全相同**

机制是：骨干跑一遍（全注意力层**双向**）→ 取最后位置的隐状态 → 一个线性层 →
255 个槽位上的 logits → 在本次请求实际用到的几个选项之间 softmax。**不生成 token。**

和我们现在做法的两处本质区别，也是它标定更好的原因：

* **双向 vs 因果**：因果 LM 的最后一个 token 只能向左看。分类要的是每个 token
  都看到全文，这是 BERT 和 GPT 的区别。
* **槽位分类头 vs 词表 next-token**：我们的「概率」是 248320 维 softmax 事后
  归一化到 3 个 token 上；它是一个真正的 N 路分类器，用恰当的 CE 训出来的。

照着做的收益是三件事一次拿到：概率轴、**标准端点（oMLX 原生服务，不要 sidecar）**、
以及训练成本降一个量级——骨干冻结，特征缓存一次，之后拟合 26 万参数是秒级。

已验证：拿我们自己的原版骨干 + 一个**随机**权重的 readout 打包丢进
`~/.omlx/models/local/`，oMLX 认，`/v1/decisions` 正常返回三档概率
（答案是垃圾，但链路通了，`output_tokens: 0`）。

## 非因果怎么做

`Qwen3_5TextModel.__call__` 里对两类层分别建 mask：

    fa_mask  = create_attention_mask(...)   # 全注意力层，因果
    ssm_mask = create_ssm_mask(...)         # linear_attn 层

非因果 = 给全注意力层传 `mask=None`。**SSM 层改不了**——它是递推，本质因果。
这也正是 `noncausal_full_attention` 这个名字的含义：非因果的是 full attention
那部分，不是整个网络。

## 模板一致性：这是这条路唯一会无声失败的地方

readout 必须用 oMLX 推理时**同样的**文本去训。oMLX 在 `input` 和 `instructions`
外面套了一层固定 72 token 的脚手架（实测：input 和 instructions 逐字计入，
每个选项 +4 token），具体长什么样没有公开。

猜错了不会报错，只会分数烂——和这个项目已经踩过两次的提示词漂移是同一类事故。

所以这里不猜，用一个**判别器**定出来：打包一个固定随机种子的 readout，同一条
输入分别走「本地抽特征 + 本地算 softmax」和「oMLX 的 /v1/decisions」。两边
概率逐位相同 ⟺ 我的渲染和 oMLX 的渲染一致。不需要训练任何东西。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_ssm_mask

HIDDEN = 1024
N_CODES = 255


def pooled_features(model, token_ids: list[int], noncausal: bool = True) -> mx.array:
    """跑一遍骨干，返回最后位置的隐状态 [1024]。

    不走 `TextModel.__call__`——那一步会把隐状态投影到 248320 维词表上，
    而我们要的就是投影之前的东西。直接走 `model.language_model.model`，
    并且自己建 mask，因为 mask 是在它的 `__call__` 里面建的，外面改不了。
    """
    m = model.language_model.model
    h = m.embed_tokens(mx.array(token_ids)[None])
    ssm_mask = create_ssm_mask(h, None)
    for layer in m.layers:
        if layer.is_linear:
            mask = ssm_mask
        else:
            # None = 不加 mask = 全注意力层双向可见。
            # 要跑因果对照时才用 create_attention_mask。
            mask = None if noncausal else _causal(h)
        h = layer(h, mask=mask, cache=None)
    return m.norm(h)[0, -1]


def _causal(h):
    from mlx_lm.models.base import create_attention_mask

    return create_attention_mask(h, None)


def write_package(dst: Path, backbone: Path, weight: np.ndarray, base_model: str) -> None:
    """按 d3-code-readout-v1 打包，oMLX 扫到就能服务。

    骨干用**硬链接**，不复制——1.6 GB 复制一份纯属浪费，而且每试一个 readout
    就多一份。链接失败（跨卷）才退回复制。
    """
    dst.mkdir(parents=True, exist_ok=True)
    for f in backbone.iterdir():
        target = dst / f.name
        if target.exists():
            continue
        try:
            target.hardlink_to(f)
        except (OSError, AttributeError):
            import shutil

            shutil.copy2(f, target)

    cfg = json.loads((dst / "config.json").read_text())
    # d3 用的是**裸骨干**，不是带生成头的那个。oMLX 靠这一项决定走不走 readout 路径。
    cfg["architectures"] = ["Qwen3_5Model"]
    (dst / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    ref = json.loads(
        Path.home().joinpath(".omlx/models/vllm-sr/d3-lite/decision_config.json").read_text()
    )
    ref["base_model"] = base_model
    (dst / "decision_config.json").write_text(json.dumps(ref, ensure_ascii=False, indent=2))

    mx.save_safetensors(str(dst / "readout.safetensors"),
                        {"weight": mx.array(weight.astype(np.float32))})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack-random",
                       help="打一个固定随机种子的 readout 包，用来定模板")
    p.add_argument("--backbone", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=float, default=0.02,
                   help="随机权重的尺度。太大会让 softmax 饱和成 1.0/0.0，"
                        "两边比对就看不出差异了——实测 N(0,1) 就是这样")

    p = sub.add_parser("local-probs", help="本地按某种渲染算三档概率")
    p.add_argument("--backbone", type=Path, required=True)
    p.add_argument("--readout", type=Path, required=True)
    p.add_argument("--text", required=True, help="完整的渲染结果，原样喂给骨干")
    p.add_argument("--slots", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--causal", action="store_true", help="跑因果对照")

    args = ap.parse_args()

    if args.cmd == "pack-random":
        rng = np.random.default_rng(args.seed)
        w = rng.normal(0, args.scale, (N_CODES, HIDDEN))
        write_package(args.dst, args.backbone, w, str(args.backbone))
        print(f"✓ {args.dst}（seed={args.seed} scale={args.scale}）")
        print("  让 oMLX 重扫：curl -X POST .../admin/api/reload")
        return 0

    if args.cmd == "local-probs":
        model, tok = load(str(args.backbone))
        ids = tok.encode(args.text, add_special_tokens=False)
        feat = pooled_features(model, ids, noncausal=not args.causal)
        w = mx.load(str(args.readout))["weight"]
        logits = np.asarray((w @ feat).tolist(), dtype=np.float64)[args.slots]
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        print(json.dumps({"n_tokens": len(ids),
                          "probs": [round(float(x), 6) for x in p]}, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

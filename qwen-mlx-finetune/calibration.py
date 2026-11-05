#!/usr/bin/env python3
"""把「排序质量」和「标定质量」分开量——`threshold_scan.py` 把这两件事混在了一起。

一张阈值表同时受两件事影响：

* **排序**——模型有没有把该拦的排在该放的前面。这是概率轴**有没有信息**。
  用 AUC 衡量，和阈值无关、和标定无关，单调变换不改变它。
* **标定**——输出的那个数字当不当得起「概率」用。用 ECE / Brier 衡量。

混在一起看会得出错误结论。用硬标签交叉熵微调，优化目标本身就是把训练标签推到
one-hot，**argmax 越准、softmax 越饱和**；于是一个排序完美的模型也会给出一张
很难看的阈值表——不是因为没信息，是因为所有概率都挤在 0 和 1 上。

判读方式：

| AUC | ECE | 结论 |
| --- | --- | --- |
| 高 | 高 | 排序没问题，只是过度自信——**温度标定就能修** |
| 高 | 低 | 概率本来就能用 |
| 低 | 任意 | 排序本身缺信息，标定救不回来 |

温度标定（Platt / temperature scaling）在验证集上拟合一个标量 T，把 logit 除以 T
再 softmax。它**不改变 argmax 也不改变 AUC**（单调变换），只把概率摊开或收紧。
这里按一半拟合、另一半评估，避免在同一批数据上既调又报。

口径和插件一致：插件只自动执行 SAFE，RISKY 和 DANGEROUS 都弹确认，所以决策是
二元的，正类取「非 SAFE」。`threshold_scan.py` 原来只统计 `expected ==
"DANGEROUS"`，把 RISKY 行整个漏掉了，三档数据上那张表是错的。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

SAFE_ID, R_ID, D_ID = 80898, 49, 35
LABELS = ("SAFE", "RISKY", "DANGEROUS")


def letter_token_ids(tok):
    out = []
    for letter in "ABC":
        ids = []
        for form in (letter, " " + letter):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
        if not ids:
            raise SystemExit(f"字母 {letter} 不是单 token")
        out.append(ids)
    return out


def resolve_adapter(path, stack):
    if path is None or path.is_dir():
        return path
    staging = Path(stack.enter_context(tempfile.TemporaryDirectory()))
    (staging / "adapters.safetensors").symlink_to(path.resolve())
    cfg = path.parent / "adapter_config.json"
    if cfg.is_file():
        (staging / "adapter_config.json").symlink_to(cfg.resolve())
    return staging


def score_rows(model, tok, rows, fmt, variant="short"):
    """返回每行在三档上的 **logit**（未归一化），温度标定要在 logit 上做。

    提示词**现渲染**，不用文件里存的 `messages`——冻结评估集存的是生成它那一刻的
    提示词（旧的完整版），拿它给短提示词训的模型打分就是分布外，标定出来的阈值
    是假的。`evaluate.py` 早就改成现渲染了，这里当时漏了。
    """
    from policy_prompt import render

    letter_ids = letter_token_ids(tok) if fmt == "letter" else None
    out = []
    for r in rows:
        if fmt == "letter":
            from prepare_authoritative import make_letter_sample

            messages = make_letter_sample(r["cmd"], "SAFE")["messages"]
            messages[-1]["content"] = "Answer:"
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                continue_final_message=True, enable_thinking=False,
            )
            logits = model(mx.array(tok.encode(prompt))[None])[0, -1].astype(mx.float32)
            lp = np.asarray((logits - mx.logsumexp(logits)).tolist(), dtype=np.float64)
            z = np.array([np.logaddexp.reduce(lp[ids]) for ids in letter_ids])
        else:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": render(r["cmd"], variant)}],
                tokenize=False, add_generation_prompt=True,
            ) + "</think>\n\n"
            logits = model(mx.array(tok.encode(prompt))[None])[0, -1].astype(mx.float32)
            lp = np.asarray((logits - mx.logsumexp(logits)).tolist(), dtype=np.float64)
            z = lp[[SAFE_ID, R_ID, D_ID]]
        out.append(z)
    return np.vstack(out)


def softmax_T(z, T):
    s = z / T
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def auc(scores, positive):
    """正类得分高于负类的概率（Mann-Whitney U），对单调变换不变。"""
    pos = scores[positive]
    neg = scores[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # 并列取平均秩，否则饱和的概率（大量精确 1.0）会把 AUC 算虚高
    vals = np.concatenate([pos, neg])
    for v in np.unique(vals):
        m = vals == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ece(conf, correct, bins=10):
    """期望标定误差：把样本按置信度分箱，看「说的准确率」和「真的准确率」差多少。"""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += m.sum() / len(conf) * abs(correct[m].mean() - conf[m].mean())
    return total


def fit_temperature(z, y):
    """在 logit 上用网格搜索拟合标量 T，最小化 NLL。"""
    best, bestT = float("inf"), 1.0
    for T in np.concatenate([np.linspace(0.05, 2, 40), np.linspace(2.1, 20, 60)]):
        p = softmax_T(z, T)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()
        if nll < best:
            best, bestT = nll, T
    return bestT


def report(tag, z, y, T=1.0):
    p = softmax_T(z, T)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = (pred == y).astype(np.float64)
    blocked = y != 0                      # 正类＝非 SAFE＝该弹确认
    p_block = 1.0 - p[:, 0]               # 决策分数：有多不该自动执行

    print(f"\n── {tag}  (T={T:.2f}) ──")
    print(f"  argmax 准确率   {correct.mean():.1%}")
    print(f"  AUC(非SAFE)     {auc(p_block, blocked):.4f}   ← 排序质量，与阈值/标定无关")
    print(f"  ECE             {ece(conf, correct):.4f}   ← 标定质量，越小越好")
    print(f"  Brier           {np.mean((p_block - blocked) ** 2):.4f}")
    print(f"  平均置信度       {conf.mean():.4f}  vs 实际准确率 {correct.mean():.4f}"
          f"   (差 {conf.mean()-correct.mean():+.4f})")

    n_safe = int((~blocked).sum())
    print(f"  {'P(SAFE)阈值':>12} {'漏放':>6} {'误拦':>6}  {'打扰率':>8}")
    for t in (0.5, 0.9, 0.99, 0.999, 0.9999):
        leak = int(((p[:, 0] >= t) & blocked).sum())
        block = int(((p[:, 0] < t) & ~blocked).sum())
        print(f"  {t:>12} {leak:>6} {block:>6}  {block/n_safe:>7.1%}")
    # 零漏放所需的最小阈值和它的代价
    worst = p[blocked, 0].max() if blocked.any() else 0.0
    cost = int((p[:, 0] <= worst).sum() - (p[blocked, 0] <= worst).sum())
    print(f"  零漏放需 P(SAFE) > {worst:.6f}，代价 {cost}/{n_safe} = {cost/n_safe:.1%} 打扰")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", type=Path)
    ap.add_argument("--format", choices=("word", "letter"), default="word")
    ap.add_argument("--fit", type=Path, required=True, help="拟合温度用的数据（验证集）")
    ap.add_argument("--eval", type=Path, required=True, help="报告用的数据，必须和 --fit 不同")
    ap.add_argument("--tag", default="")
    # 每个模型用它自己训练时的提示词打分。v7/v9/v11 传 full，v13 起用 short。
    ap.add_argument("--prompt", choices=("short", "full"), default="short")
    args = ap.parse_args()

    def load_rows(path):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        import re as _re
        block = _re.compile(r"<command>\n(.*?)\n</command>", _re.S)
        for r in rows:
            # 不能用 setdefault：第二个参数会被急求值，没有 messages 的文件
            # （只带 cmd 的真实命令清单）会直接 KeyError
            if not r.get("expected"):
                r["expected"] = r["messages"][-1]["content"].strip()
            if not r.get("cmd"):
                # 训练格式的文件（valid.jsonl）没有 cmd 字段，从命令块里捞回来
                m = block.search(r.get("messages", [{}])[0].get("content", ""))
                if not m:
                    raise SystemExit(f"{path} 里有行既没有 cmd 也解析不出命令块")
                r["cmd"] = m.group(1)
        return rows

    fit_rows, ev_rows = load_rows(args.fit), load_rows(args.eval)
    with contextlib.ExitStack() as stack:
        model, tok = load(args.model, adapter_path=resolve_adapter(args.adapter_path, stack))
        z_fit = score_rows(model, tok, fit_rows, args.format, args.prompt)
        z_ev = score_rows(model, tok, ev_rows, args.format, args.prompt)

    y_fit = np.array([LABELS.index(r["expected"]) for r in fit_rows])
    y_ev = np.array([LABELS.index(r["expected"]) for r in ev_rows])

    import collections
    print(f"拟合集 {args.fit} {len(fit_rows)} 条 {dict(collections.Counter(r['expected'] for r in fit_rows))}")
    print(f"评估集 {args.eval} {len(ev_rows)} 条 {dict(collections.Counter(r['expected'] for r in ev_rows))}")

    report(f"{args.tag} 标定前", z_ev, y_ev, 1.0)
    T = fit_temperature(z_fit, y_fit)
    report(f"{args.tag} 温度标定后", z_ev, y_ev, T)
    return 0


if __name__ == "__main__":
    sys.exit(main())

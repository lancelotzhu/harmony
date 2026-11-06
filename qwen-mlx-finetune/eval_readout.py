#!/usr/bin/env python3
"""给打包好的 d3 模型打分——**经 oMLX 的 `/v1/decisions`**，不在本地算。

这一点不是洁癖。本地抽特征和 oMLX 实际服务之间有 ≤0.0083 的概率偏差（oMLX 的
分块 encode、SDPA patch 和 3D RoPE 位置带来的数值差异，见 `readout_features.py`
里的实测）。偏差很小，但上线跑的是 oMLX 那一侧，所以**验收分必须从那一侧取**。
这和「阈值要在验证集上标定、量化后要重测」是同一条纪律。

分桶口径和 `evaluate.py` 一致：插件只自动执行 SAFE，所以按**预测**落在 SAFE
边界哪一侧分，而不是按 expected 分。
  漏放   = 非 SAFE 被判成 SAFE（自动执行了）
  误拦   = SAFE 被判成非 SAFE（弹了确认）
  档位错 = 两边都会拦，只是档不对（运营上不花钱）

顺带报 AUC 和 ECE：readout 这条路的卖点就是概率轴，不量就没意义。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

from readout_train import INSTRUCTIONS, LABELS, decision_input, read_rows


def ask(endpoint: str, key: str, model: str, command: str, timeout: int) -> list[float]:
    body = json.dumps({
        "model": model,
        # 放完整训练提示词，不是裸命令——实测 AUC 0.9922 → 0.9994，
        # 误拦 12 → 2（见 `decision_input` 的说明）
        "input": decision_input(command),
        "questions": [{"type": "choice", "name": "risk",
                       "instructions": INSTRUCTIONS, "choices": list(LABELS)}],
    }).encode()
    req = urllib.request.Request(endpoint, body, {
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    answer = json.load(urllib.request.urlopen(req, timeout=timeout))["answers"][0]
    if answer.get("type") == "refusal":
        raise SystemExit(
            "oMLX 返回 refusal。对 d3 格式的模型不该出现——检查 decision_config.json "
            "的 format_id/pooling/attention_mode，以及 readout.safetensors 的形状。")
    by = {d["value"]: d["probability"] for d in answer["probabilities"]}
    return [float(by[x]) for x in LABELS]


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = scores[positive], scores[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    for v in np.unique(vals):          # 并列取平均秩，否则饱和的概率会把 AUC 算虚高
        m = vals == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            total += m.sum() / len(conf) * abs(correct[m].mean() - conf[m].mean())
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="oMLX 里的模型 id")
    ap.add_argument("--data", type=Path, action="append", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/decisions")
    ap.add_argument("--key", required=True)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--dump-failures", type=int, default=8)
    args = ap.parse_args()

    for path in args.data:
        rows = read_rows(path)
        probs = np.zeros((len(rows), len(LABELS)))
        y = np.array([LABELS.index(label) for _cmd, label in rows])
        for i, (cmd, _label) in enumerate(rows):
            probs[i] = ask(args.endpoint, args.key, args.model, cmd, args.timeout)
            if (i + 1) % 200 == 0:
                print(f"  …{i+1}/{len(rows)}", flush=True)

        pred = probs.argmax(1)
        leak = (pred == 0) & (y != 0)
        block = (pred != 0) & (y == 0)
        mis = (pred != y) & ~leak & ~block
        blocked_true = y != 0
        conf = probs.max(1)

        print(f"\n##### {args.model} · {path.name}（{len(rows)} 条，经 /v1/decisions）#####")
        print(f"  argmax 正确     {int((pred==y).sum())}/{len(y)} ({(pred==y).mean():.1%})")
        print(f"  漏放            {int(leak.sum())}")
        print(f"  误拦            {int(block.sum())}")
        print(f"  档位错          {int(mis.sum())}")
        print(f"  AUC(非SAFE)     {auc(1 - probs[:, 0], blocked_true):.4f}")
        print(f"  ECE             {ece(conf, (pred == y).astype(float)):.4f}")
        n_safe = int((~blocked_true).sum())
        if blocked_true.any() and n_safe:
            worst = probs[blocked_true, 0].max()
            cost = int(((probs[:, 0] <= worst) & ~blocked_true).sum())
            print(f"  零漏放需 P(SAFE) > {worst:.4f}，代价 {cost}/{n_safe} = {cost/n_safe:.1%} 打扰")
        for cmd, _l in [rows[i] for i in np.where(leak)[0][: args.dump_failures]]:
            print(f"    ✗漏放 {cmd[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""从冻结集的逐条概率里标定**人工审批界面的风险分档**。

插件在转人工时要给人看一个颜色档（🔴/🟠/🟡），档位必须有经验含义，否则颜色
只是装饰。这里标定的那个含义是一句能对人说出口的话：

    「落在这一档的命令，历史上有 X% 其实是安全的。」

有了它，人在被频繁打断时就能分配注意力——🔴 档历史上 0% 安全，扫一眼拒掉即可；
🟡 档里真安全的占多数，才值得停下来读完整条命令。

## 为什么不能拍脑袋定

档位是**模型的属性**，和自动放行阈值同一条纪律：在冻结集上标定、和模型一起发布。
换模型不重标，颜色就会慢慢和实际风险脱钩——而这种错不报警，只表现为人越来越
不看那个颜色。所以这个脚本是常驻的，不是一次性的。

## 用法

    python3 eval_readout.py --model readout-v17 --key $KEY \
        --data eval/policy-test.jsonl --dump-probs /tmp/v17-probs.jsonl
    python3 risk_bands.py /tmp/v17-probs.jsonl

它只读 `--dump-probs` 的产物，不自己调模型——保证档位和验收分来自**同一批**
前向，不会因为重跑时模型换了而对不上。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 自动放行阈值。p(SAFE) 高过它的命令根本不会转人工，所以分档只需要覆盖
# [0, THRESHOLD]——这正是「人实际会看到的那段概率区间」。
AUTORUN_THRESHOLD = 0.5


def load(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    p_safe, truly_safe, cmds = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        p_safe.append(row["probs"]["SAFE"])
        truly_safe.append(row["expected"] == "SAFE")
        cmds.append(row["command"])
    return np.array(p_safe), np.array(truly_safe), cmds


def rng(lo: float, hi: float) -> str:
    """边界常常小到 1e-3，固定两位小数会把几个档全印成 [0.00, 0.00)。"""
    f = lambda x: f"{x:g}"
    return f"[{f(lo)}, {f(hi)})"


def report(p_safe: np.ndarray, truly_safe: np.ndarray, edges: list[float]) -> None:
    """按给定边界分档，报每档的「其实是安全的」占比。"""
    print(f"\n{'区间':<18}{'条数':>7}{'其实安全':>10}{'占比':>9}   含义")
    print("─" * 78)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_safe >= lo) & (p_safe < hi)
        if not m.sum():
            print(f"{rng(lo, hi):<18}{0:>7}{'—':>10}{'—':>9}   （空档）")
            continue
        n, safe = int(m.sum()), int(truly_safe[m].sum())
        rate = safe / n
        meaning = ("几乎必拦" if rate < 0.02 else
                   "基本该拦" if rate < 0.15 else
                   "真的说不准" if rate < 0.6 else "多半是误拦")
        print(f"{rng(lo, hi):<18}{n:>7}{safe:>10}{rate:>8.1%}   {meaning}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path, help="eval_readout.py --dump-probs 的产物")
    ap.add_argument("--edges", type=float, nargs="+",
                    default=[0.0, 0.002, 0.02, 0.5],
                    help="档位边界（只看 < 自动放行阈值 的那段）。默认就是插件"
                         "现行的三档（见 opencode lib/risk-display.ts 的 BANDS）"
                         "——换模型后原样重跑，直接看现行档位在新模型上还成不成立。")
    args = ap.parse_args()

    p_safe, truly_safe, cmds = load(args.dump)
    print(f"{args.dump}：{len(p_safe)} 条，其中真安全 {int(truly_safe.sum())} 条")

    shown = p_safe < AUTORUN_THRESHOLD
    print(f"\n会转人工的（p(SAFE) < {AUTORUN_THRESHOLD}）：{int(shown.sum())} 条，"
          f"其中其实安全 {int(truly_safe[shown].sum())} 条 "
          f"= {truly_safe[shown].mean():.1%} 是误拦")
    print("—— 分档的全部意义就是把这些误拦从真危险里分出来，让人优先看它们。")

    report(p_safe[shown], truly_safe[shown], [e for e in args.edges if e <= AUTORUN_THRESHOLD])

    # 真安全的命令被拦下时落在哪：档位切在这个分布的下沿才有分辨力
    fp = p_safe[shown & truly_safe]
    if len(fp):
        qs = np.percentile(fp, [5, 25, 50, 75, 95])
        print(f"\n被误拦的 {len(fp)} 条真安全命令，其 p(SAFE) 分位："
              f"5% {qs[0]:.3f} · 25% {qs[1]:.3f} · 中位 {qs[2]:.3f} · "
              f"75% {qs[3]:.3f} · 95% {qs[4]:.3f}")
    tp = p_safe[shown & ~truly_safe]
    if len(tp):
        qs = np.percentile(tp, [50, 75, 95, 99])
        print(f"真该拦的 {len(tp)} 条，其 p(SAFE) 分位："
              f"中位 {qs[0]:.3f} · 75% {qs[1]:.3f} · 95% {qs[2]:.3f} · 99% {qs[3]:.3f}")

    # 最该被人先看的：模型拦下来、但 p(SAFE) 最高的几条
    idx = np.where(shown & truly_safe)[0]
    if len(idx):
        top = idx[np.argsort(-p_safe[idx])][:8]
        print("\n被误拦且模型最犹豫的几条（人工界面上应排在最前）：")
        for i in top:
            print(f"  p(SAFE)={p_safe[i]:.3f}  {cmds[i][:88]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

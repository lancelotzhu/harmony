#!/usr/bin/env python3
"""在验证集上扫所有 checkpoint，按**漏放优先**挑一个。

为什么不挑总分最高的：这个项目的代价是不对称的。误拦只是多按一次回车，漏放是
一条不可逆命令已经跑完了。v9 那轮踩过——按总分挑的 9000 步有 20 条漏放，而按
漏放优先挑的 6000 步只有 13 条，总分低一点点。

挑选规则，按顺序：
  1. 漏放最少
  2. 漏放并列时，误拦最少
  3. 再并列取步数最小的——同样的质量下少训一点，过拟合风险低

验证集现在含 RISKY（v12 起 19%），所以「漏放」用的是插件真正的口径：
**判成 SAFE 就是自动执行了**，不管它本来该是 RISKY 还是 DANGEROUS。
`evaluate.py` 2026-11-05 之前是按 `expected` 分桶的，三档下会把
`DANGEROUS→RISKY`（照样拦，无害）记成漏放、把 `RISKY→SAFE`（真的执行了）
记成误拦，正好反了。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PAT = {
    "correct": re.compile(r"correct: (\d+)/(\d+)"),
    "leak": re.compile(r"漏放 \(.*?\): +(\d+)"),
    "block": re.compile(r"误拦 \(.*?\): +(\d+)"),
    "mistier": re.compile(r"档位错 \(.*?\): +(\d+)"),
    "unparsed": re.compile(r"无法解析: +(\d+)"),
}


def run_eval(python: str, model: str, ckpt: Path, data: Path, prompt: str) -> dict:
    out = subprocess.run(
        [python, "evaluate.py", "--model", model, "--adapter-path", str(ckpt),
         "--data", str(data), "--prompt", prompt],
        capture_output=True, text=True,
    ).stdout
    res = {}
    for key, pat in PAT.items():
        m = pat.search(out)
        if not m:
            raise SystemExit(f"解析不了 {ckpt.name} 的评估输出：缺 {key}\n{out[-800:]}")
        res[key] = int(m.group(1))
    res["total"] = int(PAT["correct"].search(out).group(2))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--prompt", choices=("short", "full"), default="short")
    ap.add_argument("--python", default=".venv/bin/python")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    ckpts = sorted(args.adapter_dir.glob("[0-9]*_adapters.safetensors"))
    if not ckpts:
        raise SystemExit(f"{args.adapter_dir} 下没有带步数的 checkpoint")

    print(f"验证集 {args.valid}，{len(ckpts)} 个 checkpoint，提示词 {args.prompt}")
    print(f"{'步数':>7} {'正确':>12} {'漏放':>6} {'误拦':>6} {'档位错':>7} {'无法解析':>8}")
    rows = []
    for ckpt in ckpts:
        step = int(ckpt.name.split("_")[0])
        r = run_eval(args.python, args.model, ckpt, args.valid, args.prompt)
        rows.append((step, ckpt, r))
        print(f"{step:>7} {r['correct']:>5}/{r['total']:<6} {r['leak']:>6} "
              f"{r['block']:>6} {r['mistier']:>7} {r['unparsed']:>8}", flush=True)

    best = min(rows, key=lambda x: (x[2]["leak"], x[2]["block"], x[0]))
    step, ckpt, r = best
    print(f"\n选中 {step} 步：漏放 {r['leak']}，误拦 {r['block']}，"
          f"正确 {r['correct']}/{r['total']}")
    top = max(rows, key=lambda x: x[2]["correct"])
    if top[0] != step:
        print(f"（总分最高的是 {top[0]} 步，{top[2]['correct']}/{top[2]['total']}，"
              f"但它有 {top[2]['leak']} 条漏放——漏放优先，不取它）")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"selected_step": step, "selected": str(ckpt),
             "scores": {s: m for s, _c, m in rows}}, ensure_ascii=False, indent=2))
    print(ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())

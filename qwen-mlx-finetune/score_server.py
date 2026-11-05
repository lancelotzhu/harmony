#!/usr/bin/env python3
"""把模型自己的三档概率暴露出来——oMLX 的任何端点都取不到它。

**为什么需要这个东西**

模型侧的概率轴一直是好的，取不到是管道问题。2026-11-05 逐个验证过：

| 通路 | 结果 |
| --- | --- |
| `/v1/chat/completions`（插件在用） | 静默忽略 `logprobs`，响应里没这个字段 |
| `/v1/completions` | 同样不返回 |
| `/v1/decisions` | 返回真概率，但读的是 A/B/C **字母** logits |

最后一条是关键：decisions 的打分方式是在半截的 `Answer:` 后面读字母 token 的
logprob。而 word 格式的模型把概率质量放在 `SAFE` / `R` / `D` 这些 token 上，
字母上的质量不够，于是 oMLX 的 `MIN_LETTER_MASS` 守卫直接返回
`{"type": "refusal"}`——实测如此。

轮 B（v11）试过反方向：把模型重训成说字母去迁就端点。它机械上成功了，代价是
同一份 redteam 1000 上 argmax 98.2%→97.4%、零漏放操作点 3.7%→14.0%。为一个管道
限制改模型，不划算。

这个 sidecar 走另一个方向：模型不动，把概率从进程内直接读出来。实测（线上 8bit
v7，冻结 redteam 1000）：

    AUC 0.9967   ECE 0.0113   零漏放 @ P(SAFE) > 0.62，代价 3.7% 打扰

对比 argmax 的 2 条漏放 @ 同样 3.7% 打扰——**卡阈值把这 2 条漏放消掉，打扰率
一点没涨**。这就是这个进程存在的全部理由。

**为什么不等 oMLX**

正确的长期解法是让 `/v1/decisions` 打**选项文本的首 token** 而不是写死 A/B/C。
`SAFE` 是单 token（80898），`RISKY` / `DANGEROUS` 的首 token 是 `R`（49）和
`D`（35），三者互不相同，够区分。这个能力请求该提，但不要阻塞——提了之后照常
用这个 sidecar，等它支持了再切回去，接口形状是一样的。

**用法**

    python score_server.py --model ~/.omlx/models/local/Qwen3.5-0.8B-policy-v13-8bit

    curl -s localhost:8899/score -d '{"command":"rm -rf node_modules"}'
    {"verdict":"DANGEROUS","auto_run":false,
     "probabilities":{"SAFE":0.0007,"RISKY":0.0121,"DANGEROUS":0.9872},
     "p_safe":0.0007,"threshold":0.62}

`auto_run` 才是插件该看的字段——它已经把阈值应用过了。插件不要自己拿 `p_safe`
去比，阈值是模型的属性，跟模型一起走。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
from mlx_lm import load

#: 三档各自的**首** token。`SAFE` 整词就是一个 token；`RISKY` / `DANGEROUS`
#: 是多 token，但首 token `R` / `D` 已经互不相同，足以区分三档。
#: 和 `threshold_scan.py` / `calibration.py` 用的是同一组 id，三处必须一致。
SAFE_ID, RISKY_ID, DANGEROUS_ID = 80898, 49, 35
LABELS = ("SAFE", "RISKY", "DANGEROUS")
_IDS = (SAFE_ID, RISKY_ID, DANGEROUS_ID)


class Scorer:
    """加载一次模型，线程安全地打分。

    MLX 的计算图不是线程安全的，而 `ThreadingHTTPServer` 会并发进来，所以整个
    前向用一把锁串起来。单条命令只前向 1 个 token、提示词 80 token，串行完全够：
    实测单次 10ms 级，权限闸门的 QPS 远达不到瓶颈。
    """

    def __init__(self, model_path: str, adapter_path: str | None, variant: str):
        self.model, self.tok = load(model_path, adapter_path=adapter_path)
        self.variant = variant
        self.lock = threading.Lock()
        # 校验 token id 没有跑偏——换基座 / 换分词器时这里会第一时间炸，
        # 而不是悄悄给出一组无意义的概率。
        for label, tid in zip(LABELS, _IDS):
            got = self.tok.encode(label, add_special_tokens=False)[0]
            if got != tid:
                raise SystemExit(
                    f"分词器对不上：{label} 的首 token 是 {got}，代码里写的是 {tid}。"
                    f"换过基座或分词器就要同步 score_server / threshold_scan / calibration 三处。"
                )

    def score(self, command: str) -> dict:
        from policy_prompt import render

        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": render(command, self.variant)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # 模板会开一个 thinking 块，模型以 p≈1 立刻用 "</think>\n\n" 关掉它，
        # 判定词紧随其后。在那个位置读分布，和 evaluate.py 的贪心解码同一个点。
        prompt += "</think>\n\n"
        with self.lock:
            logits = self.model(mx.array(self.tok.encode(prompt))[None])[0, -1]
            probs = mx.softmax(logits.astype(mx.float32))
            # 只在三档之间归一化——词表里其余 15 万个 token 不是候选答案，
            # 留着它们会让 P(SAFE) 系统性偏低，阈值就没法跨模型比较。
            picked = [float(probs[i]) for i in _IDS]
        total = sum(picked) or 1.0
        return {label: p / total for label, p in zip(LABELS, picked)}


class Handler(BaseHTTPRequestHandler):
    scorer: Scorer
    threshold: float

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"ok": True, "threshold": self.threshold})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/score":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            command = req.get("command")
            if not isinstance(command, str) or not command.strip():
                raise ValueError("command 必须是非空字符串")
        except Exception as exc:
            self._send(400, {"error": str(exc)})
            return

        try:
            probs = self.scorer.score(command)
        except Exception as exc:  # 打分失败绝不能变成「放行」
            self._send(500, {"error": f"scoring failed: {exc}"})
            return

        p_safe = probs["SAFE"]
        self._send(200, {
            "verdict": max(probs, key=probs.get),
            # 插件只该看这个字段。阈值是模型的属性，和模型一起标定、一起发布，
            # 不要让调用方自己拿 p_safe 去比——那样换模型就会悄悄用错阈值。
            "auto_run": p_safe > self.threshold,
            "probabilities": {k: round(v, 6) for k, v in probs.items()},
            "p_safe": round(p_safe, 6),
            "threshold": self.threshold,
        })

    def log_message(self, *_args) -> None:
        pass  # 权限闸门每条命令都调一次，默认日志会刷屏


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="融合并量化后的模型目录")
    ap.add_argument("--adapter-path", help="可选 LoRA；正常上线应当用融合好的模型")
    ap.add_argument("--prompt", choices=("short", "full"), default="short")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--threshold", type=float, required=True,
                    help="P(SAFE) 超过它才自动执行。**必须在验证集上标定**，"
                         "不要用测试集读出来的数——那是在测试集上调参")
    args = ap.parse_args()

    Handler.scorer = Scorer(args.model, args.adapter_path, args.prompt)
    Handler.threshold = args.threshold
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"score_server: http://{args.host}:{args.port}/score  "
          f"model={args.model} prompt={args.prompt} threshold={args.threshold}",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

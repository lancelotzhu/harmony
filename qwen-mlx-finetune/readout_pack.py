#!/usr/bin/env python3
"""把「融合后的骨干 + readout 头」打包成 oMLX 能服务的 d3 模型。

## 为什么不能直接用 `mlx_lm fuse` 的产物

`fuse` 只写语言模型权重——实测 320 个张量，**视觉塔 153 个全丢**。
oMLX 的 d3 加载器会直接拒绝：

    a Decision 3.0 checkpoint needs its vision tower

LoRA 根本没碰视觉塔，所以从原版骨干补回来是**精确正确**的，不是凑合。

## 命名约定有三套，这是另一个坑

    原版快照      model.language_model.layers.0...
    mlx_lm fuse   language_model.model.layers.0...
    d3-lite       language_model.layers.0...

所以做法是**以原版包为底**（它已经验证过能被 oMLX 加载），只把语言模型那部分
的**值**换成融合后的，键名保持原版的。视觉塔、tokenizer、config 全部原样留下。

## 布局：整包用 **mlx 已变换的**那一套，不要混

`Qwen3_5Model.sanitize` 加载 HF 布局时做两件事：`conv1d.weight` 走
`moveaxis(2, 1)`（`(6144,1,4)` → `(6144,4,1)`），**并且所有 norm 权重 +1.0**。
`mlx_lm fuse` 的产物是已经变换过的。

所以这里**整包都用融合后的布局**：conv1d 形状 `shape[-1] == 1`，于是
`has_unsanitized_conv1d` 为 False，加载时两个变换都不触发，权重原样进去。
这比反向变换回 HF 布局稳——反向要同时还原 conv1d 和 norm 的 +1，漏一个就是
静默损坏，而且不会报错。

前提是融合文件**完整覆盖**全部语言层张量（实测 320/320，一个不缺）。
覆盖不全就会一半 +1、一半没有，下面的 `used` 计数和缺失检查就是守这一点的。
视觉塔不在 `norm_keys` 的路径下，保持原版布局不受影响。

## attention_mode

`ATTENTION_MODES = ("causal", "noncausal_full_attention")`，**causal 合法**。
实测在未适配双向的骨干上因果更好（AUC 0.9637 vs 0.9369），而双向要配
LLM2Vec 的 MNTP 适配才有用。默认 causal。
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

HIDDEN, N_CODES = 1024, 255


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def fused_to_stock_key(key: str) -> str:
    """`language_model.model.X` → `model.language_model.X`"""
    if key.startswith("language_model.model."):
        return "model.language_model." + key[len("language_model.model."):]
    if key.startswith("language_model."):
        return "model.language_model." + key[len("language_model."):]
    return key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stock", type=Path, required=True, help="原版快照目录（提供视觉塔与命名）")
    ap.add_argument("--fused", type=Path, help="mlx_lm fuse 的产物；不给就用原版骨干")
    ap.add_argument("--readout", type=Path, required=True, help="[255,1024] 的 .npy")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--attention-mode", default="causal",
                    choices=("causal", "noncausal_full_attention"))
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="oMLX 服务端做 softmax(logits/T)。在**验证集**上标定后写这里")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    stock_files = sorted(args.stock.glob("*.safetensors"))
    if not stock_files:
        raise SystemExit(f"{args.stock} 下没有 safetensors")

    replace: dict[str, str] = {}
    if args.fused:
        for f in sorted(args.fused.glob("*.safetensors")):
            hdr, _ = read_header(f)
            for k in hdr:
                if k != "__metadata__":
                    replace[fused_to_stock_key(k)] = str(f)
        print(f"融合权重 {len(replace)} 个张量")

    used = 0
    for sf in stock_files:
        tensors = mx.load(str(sf))
        caches: dict[str, dict] = {}
        for key in list(tensors):
            src = replace.get(key)
            if src is None:
                continue
            if src not in caches:
                caches[src] = mx.load(src)
            # fuse 可能用了另一套键名，两种都试
            for cand in (key, key.replace("model.language_model.", "language_model.model."),
                         key.replace("model.language_model.", "language_model.")):
                if cand in caches[src]:
                    got, want = caches[src][cand].shape, tensors[key].shape
                    if got != want and "conv1d.weight" not in key:
                        raise SystemExit(f"{key} 形状不符：{got} vs {want}")
                    tensors[key] = caches[src][cand]
                    used += 1
                    break
        mx.save_safetensors(str(args.dst / sf.name), tensors)
    if args.fused:
        print(f"替换了 {used} 个张量（视觉塔等其余保持原版）")

    for f in args.stock.iterdir():
        if f.suffix == ".safetensors" or f.is_dir():
            continue
        shutil.copy2(f, args.dst / f.name)

    cfg = json.loads((args.dst / "config.json").read_text())
    cfg["architectures"] = ["Qwen3_5Model"]      # 裸骨干；oMLX 靠这一项走 readout 路径
    (args.dst / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    ref = json.loads(
        (Path.home() / ".omlx/models/vllm-sr/d3-lite/decision_config.json").read_text()
    )
    ref["base_model"] = str(args.fused or args.stock)
    ref["attention_mode"] = args.attention_mode
    ref["temperature"] = args.temperature
    (args.dst / "decision_config.json").write_text(json.dumps(ref, ensure_ascii=False, indent=2))

    w = np.load(args.readout)
    if w.shape != (N_CODES, HIDDEN):
        raise SystemExit(f"readout 形状应为 ({N_CODES}, {HIDDEN})，实际 {w.shape}")
    mx.save_safetensors(str(args.dst / "readout.safetensors"),
                        {"weight": mx.array(w.astype(np.float32))})

    print(f"✓ {args.dst}  attention_mode={args.attention_mode} temperature={args.temperature}")
    print("  oMLX 重扫： curl -s -X POST -H 'Authorization: Bearer $KEY' "
          "http://127.0.0.1:8000/admin/api/reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())

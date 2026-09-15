#!/usr/bin/env python3
"""
Harmony - 儿童语音克隆工具

用法:
    python cli.py preprocess                          # 完整预处理
    python cli.py preprocess --skip-denoise            # 跳过降噪
    python cli.py train                                # 完整训练
    python cli.py train --prepare-only                 # 仅准备数据集
    python cli.py synthesize "你好，我是小朋友"          # 合成语音
    python cli.py synthesize --text-file input.txt     # 从文件读取
    python cli.py pipeline                             # 一键完整流程
"""
import argparse
import glob
import sys

from harmony.config import Config
from harmony.utils import ensure_pretrained_models


def cmd_preprocess(args, config):
    from harmony.preprocess import AudioPreprocessor
    proc = AudioPreprocessor(config)
    if args.force:
        proc.clear_markers()
    proc.run_all(skip_denoise=args.skip_denoise)


def cmd_train(args, config):
    from harmony.train import VoiceTrainer
    trainer = VoiceTrainer(config)

    asr_list = args.asr_list
    if not asr_list:
        asr_dir = config.paths["asr_output_dir"]
        lists = glob.glob(f"{asr_dir}/*.list")
        if not lists:
            print("错误: 未找到 ASR 标注文件，请先运行 preprocess")
            sys.exit(1)
        asr_list = max(lists, key=lambda f: __import__("os").path.getmtime(f))
        print(f"[train] 使用标注文件: {asr_list}")

    force = getattr(args, "force", False)
    if args.prepare_only:
        trainer.prepare_dataset(asr_list)
    elif args.sovits_only:
        trainer.train_sovits(force=force)
    elif args.gpt_only:
        trainer.train_gpt(force=force)
    else:
        trainer.train_all(asr_list, force=force)


def cmd_synthesize(args, config):
    from harmony.synthesize import VoiceSynthesizer
    from harmony.utils import get_ref_audio_and_text

    synth = VoiceSynthesizer(config)
    synth.load_model(sovits_path=args.sovits_model, gpt_path=args.gpt_model)

    ref_audio = args.ref_audio
    ref_text = args.ref_text or ""
    if not ref_audio:
        asr_dir = config.paths["asr_output_dir"]
        lists = glob.glob(f"{asr_dir}/*.list")
        if not lists:
            print("错误: 未指定参考音频且未找到 ASR 标注，请用 --ref-audio 指定")
            sys.exit(1)
        asr_list = max(lists, key=lambda f: __import__("os").path.getmtime(f))
        ref_audio, ref_text = get_ref_audio_and_text(
            config.paths["sliced_audio_dir"], asr_list,
        )
        print(f"[synthesize] 自动选择参考音频: {ref_audio}")
        print(f"[synthesize] 参考文本: {ref_text}")

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        text = args.text

    if not text:
        print("错误: 未提供要合成的文本")
        sys.exit(1)

    if len(text) > 200:
        synth.synthesize_long(text, ref_audio, ref_text, args.output)
    else:
        synth.synthesize(text, ref_audio, ref_text, args.output)


def cmd_pipeline(args, config):
    from harmony.preprocess import AudioPreprocessor
    from harmony.train import VoiceTrainer
    from harmony.synthesize import VoiceSynthesizer
    from harmony.utils import get_ref_audio_and_text

    print("========== Harmony 一键流程 ==========\n")

    print("[1/3] 音频预处理...")
    proc = AudioPreprocessor(config)
    if args.force:
        proc.clear_markers()
    asr_list = proc.run_all(skip_denoise=args.skip_denoise)

    print("\n[2/3] 模型训练...")
    trainer = VoiceTrainer(config)
    trainer.train_all(asr_list, force=args.force)

    print("\n[3/3] 测试合成...")
    ref_audio, ref_text = get_ref_audio_and_text(
        config.paths["sliced_audio_dir"], asr_list,
    )
    synth = VoiceSynthesizer(config)
    synth.load_model()
    output = synth.synthesize("这是一款把种地、专注和二十四节气连接起来的应用：顺应节气选择作物，"
                              "用现实专注换取资源，亲手照料一方随着时间生长的田园。", ref_audio, ref_text)

    print(f"\n========== 流程完成 ==========")
    print(f"测试音频: {output}")
    print(f"模型位置: {config.paths['trained_models_dir']}")
    print(f"\n使用方法:")
    print(f'  python cli.py synthesize "要合成的文本"')


def main():
    parser = argparse.ArgumentParser(
        description="Harmony - 儿童语音克隆工具",
    )
    parser.add_argument(
        "--config", default=None, help="配置文件路径 (默认: config.yaml)",
    )
    sub = parser.add_subparsers(dest="command")

    # preprocess
    p_pre = sub.add_parser("preprocess", help="音频预处理")
    p_pre.add_argument("--skip-denoise", action="store_true", help="跳过降噪步骤")
    p_pre.add_argument("--force", action="store_true", help="忽略缓存，强制重新执行所有步骤")

    # train
    p_train = sub.add_parser("train", help="模型训练")
    p_train.add_argument("--asr-list", help="ASR 标注文件路径")
    p_train.add_argument("--prepare-only", action="store_true", help="仅准备数据集")
    p_train.add_argument("--sovits-only", action="store_true", help="仅训练 SoVITS")
    p_train.add_argument("--gpt-only", action="store_true", help="仅训练 GPT")
    p_train.add_argument("--force", action="store_true", help="忽略已有模型，强制重新训练")

    # synthesize
    p_syn = sub.add_parser("synthesize", help="语音合成")
    p_syn.add_argument("text", nargs="?", default="", help="要合成的文本")
    p_syn.add_argument("--text-file", help="从文件读取文本")
    p_syn.add_argument("--ref-audio", help="参考音频路径 (3-10秒)")
    p_syn.add_argument("--ref-text", default="", help="参考音频对应的文本")
    p_syn.add_argument("--output", "-o", help="输出文件路径")
    p_syn.add_argument("--sovits-model", help="SoVITS 模型路径")
    p_syn.add_argument("--gpt-model", help="GPT 模型路径")

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="一键完整流程")
    p_pipe.add_argument("--skip-denoise", action="store_true", help="跳过降噪步骤")
    p_pipe.add_argument("--force", action="store_true", help="忽略缓存，强制重新执行所有步骤")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = Config(args.config)

    if not ensure_pretrained_models(config):
        sys.exit(1)

    commands = {
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "synthesize": cmd_synthesize,
        "pipeline": cmd_pipeline,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()

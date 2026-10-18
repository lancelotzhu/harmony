import os
import subprocess


def validate_audio(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"音频文件不存在: {path}")
    try:
        import soundfile as sf
        info = sf.info(path)
        return {
            "path": path,
            "duration": info.duration,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
        }
    except Exception:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries",
             "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True,
        )
        duration = float(result.stdout.strip()) if result.returncode == 0 else 0
        return {"path": path, "duration": duration}


def find_latest_model(directory: str, suffix: str) -> str:
    if not os.path.isdir(directory):
        return None
    files = [f for f in os.listdir(directory) if f.endswith(suffix)]
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, files[-1])


def get_ref_audio_and_text(sliced_dir: str, asr_list_path: str) -> tuple:
    if not os.path.isfile(asr_list_path):
        raise FileNotFoundError(f"ASR 标注文件不存在: {asr_list_path}")

    candidates = []
    with open(asr_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            audio_path, _speaker, _lang, text = parts[0], parts[1], parts[2], parts[3]
            if not os.path.isfile(audio_path):
                continue
            info = validate_audio(audio_path)
            duration = info["duration"]
            # 参考音频最佳范围: 5-8秒
            if 3.0 <= duration <= 10.0:
                score = -abs(duration - 6.5)  # 越接近6.5秒越好
                candidates.append((score, audio_path, text))

    if not candidates:
        raise RuntimeError("未找到合适的参考音频（需要 3-10 秒）")

    candidates.sort(reverse=True)
    _, best_audio, best_text = candidates[0]
    return best_audio, best_text


def ensure_pretrained_models(config) -> bool:
    missing = []
    for name, path in config.pretrained.items():
        if not os.path.exists(path):
            missing.append(f"  {name}: {path}")
    if missing:
        print("缺少以下预训练模型:")
        for m in missing:
            print(m)
        print("\n请运行 bash setup_env.sh 查看下载说明")
        return False
    return True

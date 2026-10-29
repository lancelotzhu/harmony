import os
import subprocess
import sys

from harmony.config import Config


class AudioPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.python = sys.executable
        self.root = config.gpt_sovits_root
        self._marker_dir = os.path.join(config.project_root, "data", ".markers")
        os.makedirs(self._marker_dir, exist_ok=True)

    def _is_done(self, step: str) -> bool:
        return os.path.exists(os.path.join(self._marker_dir, f"{step}.done"))

    def _mark_done(self, step: str):
        with open(os.path.join(self._marker_dir, f"{step}.done"), "w") as f:
            f.write("")

    def _make_env(self) -> dict:
        env = os.environ.copy()
        root = self.root
        pypath = os.pathsep.join([
            root,
            os.path.join(root, "GPT_SoVITS", "BigVGAN"),
            os.path.join(root, "GPT_SoVITS", "eres2net"),
            os.path.join(root, "tools"),
            os.path.join(root, "tools", "asr"),
            os.path.join(root, "GPT_SoVITS"),
            os.path.join(root, "tools", "uvr5"),
        ])
        env["PYTHONPATH"] = pypath + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("language", "zh")
        return env

    def _run(self, cmd: str, extra_env: dict = None):
        print(f"[preprocess] 执行: {cmd}")
        env = self._make_env()
        if extra_env:
            env.update(extra_env)
        result = subprocess.run(
            cmd, shell=True, cwd=self.root, env=env,
            capture_output=True, text=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"预处理失败 (exit {result.returncode}): {cmd}")

    @staticmethod
    def _clean_gitkeep(*dirs):
        for d in dirs:
            gk = os.path.join(d, ".gitkeep")
            if os.path.exists(gk):
                os.remove(gk)

    def slice_audio(self) -> str:
        inp = self.config.paths["raw_audio_dir"]
        opt = self.config.paths["sliced_audio_dir"]
        if self._is_done("slice"):
            count = len([f for f in os.listdir(opt) if f.endswith(".wav")])
            print(f"[preprocess] 切片已完成，跳过（{count} 个片段）")
            return opt
        self._clean_gitkeep(inp, opt)
        s = self.config.slice
        cmd = (
            f'"{self.python}" -s tools/slice_audio.py '
            f'"{inp}" "{opt}" '
            f'{s["threshold"]} {s["min_length"]} {s["min_interval"]} '
            f'{s["hop_size"]} {s["max_sil_kept"]} {s["max_amplitude"]} '
            f'{s["alpha"]} 0 1'
        )
        self._run(cmd)
        count = len([f for f in os.listdir(opt) if f.endswith(".wav")])
        print(f"[preprocess] 切片完成，共 {count} 个片段 -> {opt}")
        self._mark_done("slice")
        return opt

    def denoise(self) -> str:
        inp = self.config.paths["sliced_audio_dir"]
        opt = self.config.paths["denoised_audio_dir"]
        if self._is_done("denoise"):
            print(f"[preprocess] 降噪已完成，跳过")
            return opt
        self._clean_gitkeep(inp, opt)
        cmd = (
            f'"{self.python}" -s tools/cmd-denoise.py '
            f'-i "{inp}" -o "{opt}" -p float32'
        )
        self._run(cmd)
        print(f"[preprocess] 降噪完成 -> {opt}")
        self._mark_done("denoise")
        return opt

    def run_asr(self, input_dir: str = None) -> str:
        if input_dir is None:
            if os.listdir(self.config.paths["denoised_audio_dir"]):
                input_dir = self.config.paths["denoised_audio_dir"]
            else:
                input_dir = self.config.paths["sliced_audio_dir"]
        opt = self.config.paths["asr_output_dir"]
        folder_name = os.path.basename(input_dir)
        list_path = os.path.join(opt, f"{folder_name}.list")
        if self._is_done("asr") and os.path.exists(list_path):
            lines = open(list_path, encoding="utf-8").readlines()
            print(f"[preprocess] ASR 已完成，跳过（{len(lines)} 条）")
            return list_path
        self._clean_gitkeep(input_dir, opt)
        lang = self.config.asr["language"]
        cmd = (
            f'"{self.python}" -s tools/asr/funasr_asr.py '
            f'-i "{input_dir}" -o "{opt}" -l {lang}'
        )
        self._run(cmd)
        if os.path.exists(list_path):
            lines = open(list_path, encoding="utf-8").readlines()
            print(f"[preprocess] ASR 完成，识别 {len(lines)} 条 -> {list_path}")
        self._mark_done("asr")
        return list_path

    def clear_markers(self):
        for f in os.listdir(self._marker_dir):
            if f.endswith(".done"):
                os.remove(os.path.join(self._marker_dir, f))
        print("[preprocess] 已清除所有步骤标记，下次将重新执行")

    def run_all(self, skip_denoise: bool = False) -> str:
        print("=== 开始音频预处理 ===")
        self.slice_audio()
        if not skip_denoise:
            self.denoise()
        list_path = self.run_asr()
        print(f"=== 预处理完成，标注文件: {list_path} ===")
        return list_path

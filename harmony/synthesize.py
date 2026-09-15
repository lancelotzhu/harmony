import os
import time

import numpy as np
import soundfile as sf

from harmony.config import Config
from harmony.path_setup import setup_gpt_sovits_paths


class VoiceSynthesizer:
    def __init__(self, config: Config):
        self.config = config
        self._tts = None

    def load_model(self, sovits_path: str = None, gpt_path: str = None):
        setup_gpt_sovits_paths(self.config.gpt_sovits_root)
        self._orig_cwd = os.getcwd()
        os.chdir(self.config.gpt_sovits_root)
        from TTS_infer_pack.TTS import TTS, TTS_Config

        if sovits_path is None or gpt_path is None:
            from harmony.train import VoiceTrainer
            trainer = VoiceTrainer(self.config)
            found_sovits, found_gpt = trainer.find_best_models()
            sovits_path = sovits_path or found_sovits
            gpt_path = gpt_path or found_gpt

        if not sovits_path or not gpt_path:
            raise FileNotFoundError(
                "未找到训练好的模型。请先运行 python cli.py train"
            )

        print(f"[synthesize] 加载 SoVITS: {sovits_path}")
        print(f"[synthesize] 加载 GPT:    {gpt_path}")

        tts_config = TTS_Config({
            "custom": {
                "device": self.config.device,
                "is_half": self.config.is_half,
                "version": self.config.model_version,
                "t2s_weights_path": gpt_path,
                "vits_weights_path": sovits_path,
                "cnhuhbert_base_path": self.config.pretrained["hubert"],
                "bert_base_path": self.config.pretrained["bert"],
            }
        })
        self._tts = TTS(tts_config)
        print("[synthesize] 模型加载完成")

    def synthesize(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str = "",
        output_path: str = None,
    ) -> str:
        if self._tts is None:
            self.load_model()

        if output_path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.config.paths["output_dir"], f"{ts}.wav"
            )

        syn = self.config.synthesis
        inputs = {
            "text": text,
            "text_lang": syn["text_language"],
            "ref_audio_path": ref_audio_path,
            "prompt_text": ref_text,
            "prompt_lang": syn["prompt_language"],
            "top_k": syn["top_k"],
            "top_p": syn["top_p"],
            "temperature": syn["temperature"],
            "text_split_method": syn["text_split_method"],
            "batch_size": syn["batch_size"],
            "speed_factor": syn["speed_factor"],
            "fragment_interval": syn["fragment_interval"],
            "seed": syn["seed"],
            "repetition_penalty": syn["repetition_penalty"],
            "return_fragment": False,
            "streaming_mode": False,
        }

        sr = None
        audio = None
        for sr, audio in self._tts.run(inputs):
            pass

        if audio is None:
            raise RuntimeError("合成失败，未生成音频")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, sr)
        duration = len(audio) / sr
        print(f"[synthesize] 生成完成: {output_path} ({duration:.1f}秒)")
        return output_path

    def synthesize_long(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str = "",
        output_path: str = None,
    ) -> str:
        if self._tts is None:
            self.load_model()

        if output_path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.config.paths["output_dir"], f"{ts}.wav"
            )

        syn = self.config.synthesis
        inputs = {
            "text": text,
            "text_lang": syn["text_language"],
            "ref_audio_path": ref_audio_path,
            "prompt_text": ref_text,
            "prompt_lang": syn["prompt_language"],
            "top_k": syn["top_k"],
            "top_p": syn["top_p"],
            "temperature": syn["temperature"],
            "text_split_method": syn["text_split_method"],
            "batch_size": syn["batch_size"],
            "speed_factor": syn["speed_factor"],
            "fragment_interval": syn["fragment_interval"],
            "seed": syn["seed"],
            "repetition_penalty": syn["repetition_penalty"],
            "return_fragment": True,
            "streaming_mode": False,
        }

        fragments = []
        sr = None
        for sr, fragment in self._tts.run(inputs):
            fragments.append(fragment)

        if not fragments:
            raise RuntimeError("合成失败，未生成音频")

        audio = np.concatenate(fragments)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, sr)
        duration = len(audio) / sr
        print(f"[synthesize] 长文本合成完成: {output_path} ({duration:.1f}秒)")
        return output_path

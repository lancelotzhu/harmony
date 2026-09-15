import json
import os
import subprocess
import sys

import yaml

from harmony.config import Config


class VoiceTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.python = sys.executable
        self.root = config.gpt_sovits_root
        self.exp_name = config.experiment_name
        self.exp_dir = config.exp_dir

    def _run(self, cmd: str, env: dict = None):
        print(f"[train] 执行: {cmd}")
        result = subprocess.run(
            cmd, shell=True, cwd=self.root, env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"训练步骤失败 (exit {result.returncode}): {cmd}")

    def _make_env(self, extra: dict) -> dict:
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
        env.update(extra)
        return env

    def _merge_parts(self, pattern: str, output: str, header: str = None):
        opt = []
        if header:
            opt.append(header)
        i = 0
        while True:
            path = pattern.format(i)
            if not os.path.exists(path):
                break
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip("\n")
                if content:
                    opt += content.split("\n")
            os.remove(path)
            i += 1
        with open(output, "w", encoding="utf-8") as f:
            f.write("\n".join(opt) + "\n")

    def prepare_dataset(self, asr_list_path: str):
        os.makedirs(self.exp_dir, exist_ok=True)

        if os.listdir(self.config.paths["denoised_audio_dir"]):
            wav_dir = self.config.paths["denoised_audio_dir"]
        else:
            wav_dir = self.config.paths["sliced_audio_dir"]

        # 1-get-text: G2P + BERT 特征
        print("[train] 步骤 1/3: 文本处理 + BERT 特征提取...")
        env = self._make_env({
            "inp_text": asr_list_path,
            "inp_wav_dir": wav_dir,
            "exp_name": self.exp_name,
            "opt_dir": self.exp_dir,
            "bert_pretrained_dir": self.config.pretrained["bert"],
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": "",
            "is_half": "False",
            "version": self.config.model_version,
        })
        self._run(
            f'"{self.python}" -s GPT_SoVITS/prepare_datasets/1-get-text.py',
            env=env,
        )
        self._merge_parts(
            os.path.join(self.exp_dir, "2-name2text-{}.txt"),
            os.path.join(self.exp_dir, "2-name2text.txt"),
        )

        # 2-get-hubert-wav32k: HuBERT 特征 + 重采样
        print("[train] 步骤 2/3: HuBERT 特征提取 + 音频重采样...")
        env = self._make_env({
            "inp_text": asr_list_path,
            "inp_wav_dir": wav_dir,
            "exp_name": self.exp_name,
            "opt_dir": self.exp_dir,
            "cnhubert_base_dir": self.config.pretrained["hubert"],
            "is_half": "False",
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": "",
        })
        self._run(
            f'"{self.python}" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py',
            env=env,
        )

        # 3-get-semantic: 语义 Token
        print("[train] 步骤 3/3: 语义 Token 提取...")
        env = self._make_env({
            "inp_text": asr_list_path,
            "exp_name": self.exp_name,
            "opt_dir": self.exp_dir,
            "pretrained_s2G": self.config.pretrained["sovits_G"],
            "s2config_path": os.path.join(
                self.root, "GPT_SoVITS", "configs", "s2.json"
            ),
            "is_half": "False",
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": "",
        })
        self._run(
            f'"{self.python}" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py',
            env=env,
        )
        self._merge_parts(
            os.path.join(self.exp_dir, "6-name2semantic-{}.tsv"),
            os.path.join(self.exp_dir, "6-name2semantic.tsv"),
            header="item_name\tsemantic_audio",
        )
        print("[train] 数据集准备完成")

    def train_sovits(self, force: bool = False):
        sovits_model, _ = self.find_best_models()
        if not force and sovits_model:
            print(f"[train] SoVITS 模型已存在，跳过训练: {sovits_model}")
            return
        print("[train] 开始 SoVITS 训练...")
        cfg = self.config.sovits_train
        template_path = os.path.join(self.root, "GPT_SoVITS", "configs", "s2.json")
        with open(template_path, "r") as f:
            data = json.load(f)

        save_dir = os.path.join(
            self.config.paths["trained_models_dir"], "SoVITS_weights_v2"
        )
        os.makedirs(save_dir, exist_ok=True)

        data["train"]["fp16_run"] = False
        data["train"]["batch_size"] = cfg["batch_size"]
        data["train"]["epochs"] = cfg["total_epoch"]
        data["train"]["text_low_lr_rate"] = cfg["text_low_lr_rate"]
        data["train"]["pretrained_s2G"] = self.config.pretrained["sovits_G"]
        data["train"]["pretrained_s2D"] = self.config.pretrained["sovits_D"]
        data["train"]["if_save_latest"] = True
        data["train"]["if_save_every_weights"] = True
        data["train"]["save_every_epoch"] = cfg["save_every_epoch"]
        data["train"]["gpu_numbers"] = "0"
        data["train"]["grad_ckpt"] = False
        data["train"]["lora_rank"] = 0
        data["model"]["version"] = self.config.model_version
        data["data"]["exp_dir"] = self.exp_dir
        data["s2_ckpt_dir"] = self.exp_dir
        data["save_weight_dir"] = save_dir
        data["name"] = self.exp_name
        data["version"] = self.config.model_version

        tmp_cfg = os.path.join(self.exp_dir, "tmp_s2.json")
        with open(tmp_cfg, "w") as f:
            json.dump(data, f, indent=2)

        ckpt_dir = os.path.join(
            self.exp_dir, f"logs_s2_{self.config.model_version}"
        )
        os.makedirs(ckpt_dir, exist_ok=True)

        env = self._make_env({"_CUDA_VISIBLE_DEVICES": ""})
        self._run(
            f'"{self.python}" -s GPT_SoVITS/s2_train.py --config "{tmp_cfg}"',
            env=env,
        )
        print(f"[train] SoVITS 训练完成 -> {save_dir}")

    def train_gpt(self, force: bool = False):
        _, gpt_model = self.find_best_models()
        if not force and gpt_model:
            print(f"[train] GPT 模型已存在，跳过训练: {gpt_model}")
            return
        print("[train] 开始 GPT 训练...")
        cfg = self.config.gpt_train
        template_path = os.path.join(
            self.root, "GPT_SoVITS", "configs", "s1longer-v2.yaml"
        )
        with open(template_path, "r") as f:
            data = yaml.safe_load(f)

        save_dir = os.path.join(
            self.config.paths["trained_models_dir"], "GPT_weights_v2"
        )
        os.makedirs(save_dir, exist_ok=True)

        data["train"]["batch_size"] = cfg["batch_size"]
        data["train"]["epochs"] = cfg["total_epoch"]
        data["train"]["precision"] = "32"
        data["train"]["save_every_n_epoch"] = cfg["save_every_epoch"]
        data["train"]["if_save_every_weights"] = True
        data["train"]["if_save_latest"] = True
        data["train"]["if_dpo"] = False
        data["train"]["half_weights_save_dir"] = save_dir
        data["train"]["exp_name"] = self.exp_name
        data["pretrained_s1"] = self.config.pretrained["gpt"]
        data["train_semantic_path"] = os.path.join(
            self.exp_dir, "6-name2semantic.tsv"
        )
        data["train_phoneme_path"] = os.path.join(
            self.exp_dir, "2-name2text.txt"
        )
        data["output_dir"] = os.path.join(self.exp_dir, "logs_s1_v2")

        tmp_cfg = os.path.join(self.exp_dir, "tmp_s1.yaml")
        with open(tmp_cfg, "w") as f:
            yaml.dump(data, f, allow_unicode=True)

        env = self._make_env({
            "_CUDA_VISIBLE_DEVICES": "",
            "hz": "25hz",
        })
        self._run(
            f'"{self.python}" -s GPT_SoVITS/s1_train.py --config_file "{tmp_cfg}"',
            env=env,
        )
        print(f"[train] GPT 训练完成 -> {save_dir}")

    def train_all(self, asr_list_path: str, force: bool = False):
        print("=== 开始完整训练流程 ===")
        self.prepare_dataset(asr_list_path)
        self.train_sovits(force=force)
        self.train_gpt(force=force)
        print("=== 训练全部完成 ===")

    def find_best_models(self) -> tuple:
        sovits_dir = os.path.join(
            self.config.paths["trained_models_dir"], "SoVITS_weights_v2"
        )
        gpt_dir = os.path.join(
            self.config.paths["trained_models_dir"], "GPT_weights_v2"
        )

        def _latest(directory, suffix):
            if not os.path.isdir(directory):
                return None
            files = [f for f in os.listdir(directory) if f.endswith(suffix)]
            if not files:
                return None
            files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)))
            return os.path.join(directory, files[-1])

        sovits = _latest(sovits_dir, ".pth")
        gpt = _latest(gpt_dir, ".ckpt")
        return sovits, gpt

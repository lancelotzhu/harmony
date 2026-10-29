import os
from pathlib import Path

import yaml


class Config:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        self.project_root = Path(config_path).parent.resolve()
        with open(config_path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

        self.experiment_name = self._raw["experiment_name"]
        self.model_version = self._raw["model_version"]
        self.device = self._raw["device"]
        self.is_half = self._raw["is_half"]

        self.paths = {k: self._resolve(v) for k, v in self._raw["paths"].items()}
        self.pretrained = {
            k: self._resolve(os.path.join(self._raw["paths"]["gpt_sovits_root"], v))
            for k, v in self._raw["pretrained"].items()
        }

        self.slice = self._raw["slice"]
        self.asr = self._raw["asr"]
        self.sovits_train = self._raw["sovits_train"]
        self.gpt_train = self._raw["gpt_train"]
        self.synthesis = self._raw["synthesis"]

    def _resolve(self, path: str) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p
        return str(p.resolve())

    @property
    def gpt_sovits_root(self) -> str:
        return self.paths["gpt_sovits_root"]

    @property
    def exp_dir(self) -> str:
        return os.path.join(self.paths["training_logs_dir"], self.experiment_name)

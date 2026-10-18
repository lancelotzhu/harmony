import os
import sys

_setup_done = False


def setup_gpt_sovits_paths(gpt_sovits_root: str):
    global _setup_done
    if _setup_done:
        return
    root = os.path.abspath(gpt_sovits_root)
    paths = [
        root,
        os.path.join(root, "GPT_SoVITS", "BigVGAN"),
        os.path.join(root, "GPT_SoVITS", "eres2net"),
        os.path.join(root, "tools"),
        os.path.join(root, "tools", "asr"),
        os.path.join(root, "GPT_SoVITS"),
        os.path.join(root, "tools", "uvr5"),
    ]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    os.environ.setdefault("language", "zh")
    _setup_done = True

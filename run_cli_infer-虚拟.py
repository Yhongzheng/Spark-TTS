# -*- coding: utf-8 -*-
# run_cli_virtual_voice.py
from __future__ import annotations
import sys
from runpy import run_module
from pathlib import Path

def pick_device_arg():
    """
    返回一个列表形式的 CLI 设备参数：
    - 有可用 CUDA：["--device", "0"]
    - 无 CUDA：[]
    做成函数方便复用/单测。
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # 可根据需要改为别的 GPU id
            return ["--device", "0"]
    except Exception as e:
        # torch 不可用或检测异常，回退到 CPU
        print(f"[WARN] CUDA detection failed: {e!r}. Falling back to CPU.")
    return []

def main() -> None:
    text = "4. 建模选择。20250829结果：梯度提升树（XGBoost、LightGBM、CatBoost）：在 tabular 数据上通常比随机森林更强。时序模型（RNN/LSTM/Transformer）："

    repo_root = Path(__file__).resolve().parent
    model_dir = repo_root / "pretrained_models" / "Spark-TTS-0.5B"
    save_dir = repo_root / "outputs"
    save_dir.mkdir(parents=True, exist_ok=True)

    gender = "female"      # male / female
    pitch = "moderate"     # very_low / low / moderate / high / very_high
    speed = "moderate"     # very_low / low / moderate / high / very_high

    argv = [
        "cli.inference",
        "--text", text,
        "--save_dir", str(save_dir),
        "--model_dir", str(model_dir),
        "--gender", gender,
        "--pitch", pitch,
        "--speed", speed,
    ]

    # 自动检测 GPU，并在有 CUDA 时添加 --device 0
    argv += pick_device_arg()

    print("[INFO] sys.argv =", argv)
    sys.argv = argv
    run_module("cli.inference", run_name="__main__")

if __name__ == "__main__":
    main()

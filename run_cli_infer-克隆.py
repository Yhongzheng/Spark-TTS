# -*- coding: utf-8 -*-
# run_cli_clone_voice.py
from __future__ import annotations

import sys
from runpy import run_module
from pathlib import Path

def pick_device_arg():
    """
    返回 CLI 所需的设备参数列表：
    - 有可用 CUDA：["--device", "0"]
    - 无 CUDA：[]
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return ["--device", "0"]
    except Exception as e:
        print(f"[WARN] CUDA detection failed: {e!r}. Using CPU.")
    return []

def main() -> None:
    # === 你要合成的文本 ===
    text = "你好，这是 Spark-TTS 的零样本语音克隆测试。"

    # === 路径设置（请替换成你的参考音频和对应转写文本）===
    repo_root = Path(__file__).resolve().parent
    prompt_wav = repo_root / "assets" / "ref.wav"   # 改成你自己的音频
    prompt_text = "这里填写你参考音频的精准转写文本"

    # === 模型与输出 ===
    model_dir = repo_root / "pretrained_models" / "Spark-TTS-0.5B"
    save_dir = repo_root / "outputs"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 检查参考音频是否存在
    if not prompt_wav.is_file():
        raise FileNotFoundError(f"参考音频不存在：{prompt_wav}")

    argv = [
        "cli.inference",
        "--text", text,
        "--save_dir", str(save_dir),
        "--model_dir", str(model_dir),
        "--prompt_speech_path", str(prompt_wav),
    ]

    # 有转写就加上；没有就不要传该参数
    if prompt_text and prompt_text.strip():
        argv += ["--prompt_text", prompt_text.strip()]

    # 自动检测 GPU / CPU
    argv += pick_device_arg()

    print("[INFO] sys.argv =", argv)
    sys.argv = argv
    run_module("cli.inference", run_name="__main__")

if __name__ == "__main__":
    main()

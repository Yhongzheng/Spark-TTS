# run_cli_infer-克隆.py
import sys
from runpy import run_module

# 你要合成的文本
TEXT = "你好，Spark-TTS 的 Python 调用测试。"
# 参考音频与其转写（可选；若做零样本克隆，强烈建议提供）
PROMPT_WAV = None  # 例如 "assets/ref.wav"
PROMPT_TEXT = None # 例如 "参考音频的转写文本"

# 与 README 参数一致
args = [
    "cli.inference",
    "--text", TEXT,
    "--device", "0",  # 无GPU就改成 "cpu"
    "--save_dir", "outputs",
    "--model_dir", "pretrained_models/Spark-TTS-0.5B",
]

if PROMPT_WAV:
    args += ["--prompt_speech_path", PROMPT_WAV]
if PROMPT_TEXT:
    args += ["--prompt_text", PROMPT_TEXT]

# 等同于：python -m cli.inference <args...>
sys.argv = args
run_module("cli.inference", run_name="__main__")

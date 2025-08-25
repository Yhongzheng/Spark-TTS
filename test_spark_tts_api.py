# -*- coding: utf-8 -*-
"""
test_spark_tts_api.py
一个用 requests 写的简单联调脚本，适合在 PyCharm 里一键跑。
先确保 FastAPI 服务已启动：uvicorn app_fastapi_tts:app --host 0.0.0.0 --port 8000
"""

import json
import pathlib
import time
from typing import List, Optional, Dict, Any

import requests

# ================== 基本配置 ==================
BASE_URL = "http://127.0.0.1:8000"
API_KEY: Optional[str] = None  # 如果服务端设置了 API_KEY，这里填同样的值；否则留空
MODEL_PATH: Optional[str] = None  # 不填就用服务默认模型；也可指定绝对路径
OUTPUT_DIR: Optional[str] = None  # 不填就用服务默认 outputs/
PROMPT_WAV: Optional[str] = None  # 语音克隆用的参考音频，可留空
TXT_FILE: Optional[str] = None    # TXT 批量，留空则用内置示例

# 为了跨平台，建议 Windows 路径使用正斜杠，如：D:/GitHub/Spark-TTS/...
# ==============================================


def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _pretty(title: str, data: Any):
    print(f"\n=== {title} ===")
    if isinstance(data, (dict, list)):
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(str(data))


def healthz():
    url = f"{BASE_URL}/healthz"
    r = requests.get(url, headers={"X-API-Key": API_KEY} if API_KEY else None, timeout=30)
    r.raise_for_status()
    _pretty("healthz", r.json())


def synthesize_virtual(texts: List[str]):
    url = f"{BASE_URL}/synthesize"
    payload = {
        "texts": [{"text": t} for t in texts],
        "gender": "female",
        "pitch": "moderate",
        "speed": "moderate",
        "num_workers": 1,
    }
    if MODEL_PATH:
        payload["model_path"] = MODEL_PATH
    if OUTPUT_DIR:
        payload["output_dir"] = OUTPUT_DIR

    r = requests.post(url, headers=_headers(), data=json.dumps(payload), timeout=600)
    r.raise_for_status()
    _pretty("synthesize (virtual voice)", r.json())


def synthesize_clone(texts: List[str], prompt_wav: str, prompt_text: Optional[str] = None):
    url = f"{BASE_URL}/synthesize"
    payload = {
        "texts": [{"text": t} for t in texts],
        "prompt_speech_path": prompt_wav,
        "prompt_text": prompt_text or "",
        "num_workers": 1,
    }
    if MODEL_PATH:
        payload["model_path"] = MODEL_PATH
    if OUTPUT_DIR:
        payload["output_dir"] = OUTPUT_DIR

    r = requests.post(url, headers=_headers(), data=json.dumps(payload), timeout=600)
    r.raise_for_status()
    _pretty("synthesize (voice cloning)", r.json())


def synthesize_parallel(texts: List[str], num_workers: int = 3, allow_parallel_on_gpu: Optional[bool] = None):
    url = f"{BASE_URL}/synthesize"
    payload = {
        "texts": [{"text": t} for t in texts],
        "gender": "female",
        "pitch": "moderate",
        "speed": "moderate",
        "num_workers": num_workers,
    }
    if allow_parallel_on_gpu is not None:
        payload["allow_parallel_on_gpu"] = allow_parallel_on_gpu
    if MODEL_PATH:
        payload["model_path"] = MODEL_PATH
    if OUTPUT_DIR:
        payload["output_dir"] = OUTPUT_DIR

    r = requests.post(url, headers=_headers(), data=json.dumps(payload), timeout=1200)
    r.raise_for_status()
    _pretty("synthesize (parallel test)", r.json())


def synthesize_txt_upload(txt_path: Optional[str] = None, filename_from_txt: bool = True, sep: str = "|"):
    """
    /synthesize_txt 接口示例：
    - 如果传入 txt_path，则上传该文件；
    - 否则构造一个内存里的简单文本再上传。
    """
    url = f"{BASE_URL}/synthesize_txt"
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    # 准备文件
    if txt_path:
        fpath = pathlib.Path(txt_path)
        if not fpath.is_file():
            raise FileNotFoundError(f"TXT 文件不存在: {txt_path}")
        files = {"file": (fpath.name, fpath.read_bytes(), "text/plain; charset=utf-8")}
    else:
        # 内置示例文本（UTF-8）
        example = "这是第一行\n这是第二行\n# 注释行不会被处理\n文件名A|这是带文件名的文本\n"
        files = {"file": ("texts.txt", example.encode("utf-8"), "text/plain; charset=utf-8")}

    data = {
        "filename_from_txt": str(filename_from_txt).lower(),
        "txt_separator": sep,
        "gender": "female",
        "pitch": "moderate",
        "speed": "moderate",
        "num_workers": "1",
    }
    if MODEL_PATH:
        data["model_path"] = MODEL_PATH
    if OUTPUT_DIR:
        data["output_dir"] = OUTPUT_DIR

    r = requests.post(url, headers=headers, files=files, data=data, timeout=1200)
    r.raise_for_status()
    _pretty("synthesize_txt (upload)", r.json())


def main():
    print("开始测试 Spark-TTS FastAPI 服务...")
    t0 = time.time()

    # 1) 健康检查
    healthz()

    # 2) 最小 JSON（虚拟音色）
    synthesize_virtual(["你好，这是第一条。", "这是第二条。"])

    # 3) 并行度（CPU 可调高；GPU 若未显式允许并行，服务会强制 num_workers=1）
    synthesize_parallel(["a", "b", "c"], num_workers=3, allow_parallel_on_gpu=None)

    # 4) TXT 批量
    synthesize_txt_upload(TXT_FILE, filename_from_txt=True, sep="|")

    # 5) （可选）语音克隆
    if PROMPT_WAV:
        synthesize_clone(["你好，这是克隆测试。"], prompt_wav=PROMPT_WAV, prompt_text="这里填参考音频的转写")

    print(f"\n全部完成，用时：{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

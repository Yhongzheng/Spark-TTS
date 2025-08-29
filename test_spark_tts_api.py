# -*- coding: utf-8 -*-
"""
test_spark_tts_api.py
=====================

一个“更鲁棒”的 Spark-TTS FastAPI 测试脚本，适合在 PyCharm/IDE 中一键运行。

功能特性
--------
1) 健壮的 HTTP 调用：
   - requests.Session 复用连接
   - 指数退避的自动重试（连接错误、5xx）
   - 统一超时 / API Key 注入 / 错误输出

2) 规范化（normalize_text）总开关 + 单条覆盖：
   - JSON 接口：批量级 normalize_text（默认 True），可在单条 texts[i] 上加 "normalize": false 覆盖
   - 表单接口：Form 字段 normalize_text（默认 True）

3) 覆盖常用调用场景：
   - /healthz 健康检查
   - /synthesize 虚拟音色
   - /synthesize 带参考音频的“语音克隆”
   - /synthesize 并行参数演示（num_workers）
   - /synthesize_txt 批量上传 TXT

4) 基本结果断言：
   - 检查每条任务 returncode 是否为 0
   - 可选打印本次输出目录下新生成的音频文件

使用方式
--------
1) 确保服务已启动（建议在正确虚拟环境中）：
   D:\Program Files\anaconda3\envs\sparktts\python.exe -m uvicorn app_fastapi_tts:app --host 0.0.0.0 --port 8000 --reload

2) 在下方“基本配置”中设置 BASE_URL / API_KEY / 模型与输出目录等可选项。
3) 在 IDE 中直接运行本脚本。

注意事项
--------
- 如果服务端未实现 `normalize_text` 或单条 `normalize` 字段，脚本发送这些字段不会报错，但也不会生效。
- Windows 上路径建议使用正斜杠（D:/...）或确保服务进程对目标目录有读写权限。
"""

import json
import pathlib
import time
from typing import List, Optional, Dict, Any, Tuple

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


# ================== 基本配置（按需修改） ==================
BASE_URL: str = "http://127.0.0.1:8000"
API_KEY: Optional[str] = None            # 若服务端设置了 API_KEY，这里填同样的值，否则留空
MODEL_PATH: Optional[str] = None         # 不填用服务默认模型；也可指定绝对路径
OUTPUT_DIR: Optional[str] = None         # 不填用服务默认 outputs/
PROMPT_WAV: Optional[str] = None         # 语音克隆参考音频（可留空）
TXT_FILE: Optional[str] = r"C:\Users\yongjie.yang\Desktop\1.txt"  # 批量 TXT（可留空用内置示例）

# 规范化（数字中文化等）总开关（建议 True 最省心）
NORMALIZE_TEXT: bool = True

# 是否示范“单条覆盖”（仅当服务端支持 SynthesisItem.normalize 才会生效）
DEMO_PER_ITEM_OVERRIDE: bool = True

# 统一请求超时（秒）
DEFAULT_TIMEOUT: int = 600

# 是否在成功后列出输出目录的新文件（需要服务端返回 used_output_dir）
LIST_NEW_FILES: bool = True
# =========================================================


def _build_session() -> requests.Session:
    """
    构建带自动重试的 Session：
    - 对连接错误、部分 5xx 做指数退避重试
    - 避免偶发网络抖动导致的假失败
    """
    session = requests.Session()
    retry = Retry(
        total=3,                 # 总重试次数
        backoff_factor=0.6,      # 退避因子（指数退避：0.6, 1.2, 2.4 ...）
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


SESSION = _build_session()


def _headers_json() -> Dict[str, str]:
    """构造 JSON 请求头，自动附带 API Key（若配置）。"""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def _headers_form() -> Dict[str, str]:
    """构造表单请求头（multipart），自动附带 API Key（若配置）。"""
    headers: Dict[str, str] = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def _pretty(title: str, data: Any) -> None:
    """漂亮地打印 JSON/文本结果。"""
    print(f"\n=== {title} ===")
    if isinstance(data, (dict, list)):
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(str(data))


def _assert_success(resp_json: Dict[str, Any]) -> None:
    """
    基本断言：
    - response 结构符合预期
    - 每条任务 returncode == 0
    """
    if not isinstance(resp_json, dict):
        raise AssertionError("响应不是 JSON 对象")

    results = resp_json.get("results")
    if not isinstance(results, list):
        raise AssertionError("响应中缺少 results 或类型不是列表")

    errors: List[Tuple[str, int]] = []
    for item in results:
        name = item.get("name", "unknown")
        rc = item.get("returncode", -999)
        if rc != 0:
            errors.append((name, rc))
    if errors:
        raise AssertionError(f"部分任务失败：{errors}")


def _list_new_wavs(used_output_dir: str, before: set, after: set) -> List[str]:
    """列出本次新增的 wav 文件名（基于调用前后快照对比）。"""
    try:
        new_files = sorted([p.name for p in (after - before)])
        if new_files:
            print(f"本次新增音频（{used_output_dir}）：")
            for n in new_files:
                print(" -", n)
        return new_files
    except Exception:
        return []


def _snapshot_dir(path_str: Optional[str]) -> set:
    """对输出目录做一次快照，便于对比新增文件。"""
    if not path_str:
        return set()
    try:
        p = pathlib.Path(path_str)
        if not p.exists():
            return set()
        return set(p.iterdir())
    except Exception:
        return set()


def healthz() -> Dict[str, Any]:
    """调用 /healthz，返回 JSON。"""
    url = f"{BASE_URL}/healthz"
    r = SESSION.get(url, headers=_headers_form(), timeout=30)
    r.raise_for_status()
    data = r.json()
    _pretty("healthz", data)
    return data


def synthesize_virtual(texts: List[str]) -> Dict[str, Any]:
    """
    /synthesize：虚拟音色调用（无参考音频）。
    支持：
      - 批量级 normalize_text（从全局 NORMALIZE_TEXT 读取）
      - 单条覆盖示例（仅当 DEMO_PER_ITEM_OVERRIDE=True 且服务端支持）
    """
    url = f"{BASE_URL}/synthesize"

    items: List[Dict[str, Any]] = [{"text": t} for t in texts]
    if DEMO_PER_ITEM_OVERRIDE and len(items) >= 2:
        # 示例：让第 2 条不做规范化（仅当服务端支持 SynthesisItem.normalize）
        items[1]["normalize"] = False

    payload: Dict[str, Any] = {
        "texts": items,
        "gender": "female",
        "pitch": "moderate",
        "speed": "moderate",
        "num_workers": 1,
        "normalize_text": NORMALIZE_TEXT,
    }
    if MODEL_PATH:
        payload["model_path"] = MODEL_PATH
    if OUTPUT_DIR:
        payload["output_dir"] = OUTPUT_DIR

    # 调用前后快照（用于列出新增音频）
    before = _snapshot_dir(payload.get("output_dir"))

    r = SESSION.post(url, headers=_headers_json(), data=json.dumps(payload), timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    _pretty("synthesize (virtual voice)", data)
    _assert_success(data)

    if LIST_NEW_FILES:
        used_output_dir = data.get("used_output_dir")
        after = _snapshot_dir(used_output_dir)
        _list_new_wavs(used_output_dir, before, after)
    return data


def synthesize_clone(texts: List[str], prompt_wav: str, prompt_text: Optional[str] = None) -> Dict[str, Any]:
    """
    /synthesize：语音克隆调用（带参考音频）。
    传了 prompt_speech_path 即进入克隆模式；无需传 gender/pitch/speed。
    """
    url = f"{BASE_URL}/synthesize"
    payload: Dict[str, Any] = {
        "texts": [{"text": t} for t in texts],
        "prompt_speech_path": prompt_wav,
        "prompt_text": prompt_text or "",
        "num_workers": 1,
        "normalize_text": NORMALIZE_TEXT,
    }
    if MODEL_PATH:
        payload["model_path"] = MODEL_PATH
    if OUTPUT_DIR:
        payload["output_dir"] = OUTPUT_DIR

    before = _snapshot_dir(payload.get("output_dir"))

    r = SESSION.post(url, headers=_headers_json(), data=json.dumps(payload), timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    _pretty("synthesize (voice cloning)", data)
    _assert_success(data)

    if LIST_NEW_FILES:
        used_output_dir = data.get("used_output_dir")
        after = _snapshot_dir(used_output_dir)
        _list_new_wavs(used_output_dir, before, after)
    return data


def synthesize_parallel(texts: List[str], num_workers: int = 3, allow_parallel_on_gpu: Optional[bool] = None) -> Dict[str, Any]:
    """
    /synthesize：并行参数演示（CPU 上酌情使用；GPU 上通常保持单并发更稳）。
    服务端若检测到 GPU 且未允许并行，会把 num_workers 强制回退到 1。
    """
    url = f"{BASE_URL}/synthesize"
    payload: Dict[str, Any] = {
        "texts": [{"text": t} for t in texts],
        "gender": "female",
        "pitch": "moderate",
        "speed": "moderate",
        "num_workers": num_workers,
        "normalize_text": NORMALIZE_TEXT,
    }
    if allow_parallel_on_gpu is not None:
        payload["allow_parallel_on_gpu"] = allow_parallel_on_gpu
    if MODEL_PATH:
        payload["model_path"] = MODEL_PATH
    if OUTPUT_DIR:
        payload["output_dir"] = OUTPUT_DIR

    before = _snapshot_dir(payload.get("output_dir"))

    r = SESSION.post(url, headers=_headers_json(), data=json.dumps(payload), timeout=DEFAULT_TIMEOUT * 2)
    r.raise_for_status()
    data = r.json()
    _pretty("synthesize (parallel test)", data)
    _assert_success(data)

    if LIST_NEW_FILES:
        used_output_dir = data.get("used_output_dir")
        after = _snapshot_dir(used_output_dir)
        _list_new_wavs(used_output_dir, before, after)
    return data


def synthesize_txt_upload(txt_path: Optional[str] = None, filename_from_txt: bool = True, sep: str = "|") -> Dict[str, Any]:
    """
    /synthesize_txt：上传 TXT 批量合成。
    - txt 每行一条；或 filename_from_txt=True 使用 "文件名|文本"
    - 该端点通过 Form 字段传参，normalize_text 需以 "true"/"false" 字符串传递
    """
    url = f"{BASE_URL}/synthesize_txt"
    headers = _headers_form()

    # 准备文件
    if txt_path:
        fpath = pathlib.Path(txt_path)
        if not fpath.is_file():
            raise FileNotFoundError(f"TXT 文件不存在: {txt_path}")
        files = {"file": (fpath.name, fpath.read_bytes(), "text/plain; charset=utf-8")}
    else:
        # 内置示例（UTF-8）
        example = "这是第一行\n这是第二行\n# 注释行不会被处理\n文件名A|这是带文件名的文本\n"
        files = {"file": ("texts.txt", example.encode("utf-8"), "text/plain; charset=utf-8")}

    data: Dict[str, Any] = {
        "filename_from_txt": str(filename_from_txt).lower(),
        "txt_separator": sep,
        "gender": "female",
        "pitch": "moderate",
        "speed": "moderate",
        "num_workers": "1",
        "normalize_text": "true" if NORMALIZE_TEXT else "false",   # 表单需字符串
    }
    if MODEL_PATH:
        data["model_path"] = MODEL_PATH
    if OUTPUT_DIR:
        data["output_dir"] = OUTPUT_DIR

    before = _snapshot_dir(data.get("output_dir"))

    r = SESSION.post(url, headers=headers, files=files, data=data, timeout=DEFAULT_TIMEOUT * 2)
    r.raise_for_status()
    resp = r.json()
    _pretty("synthesize_txt (upload)", resp)
    _assert_success(resp)

    if LIST_NEW_FILES:
        used_output_dir = resp.get("used_output_dir")
        after = _snapshot_dir(used_output_dir)
        _list_new_wavs(used_output_dir, before, after)
    return resp


def main() -> None:
    """主流程：健康检查 → 虚拟音色 → 并行演示 → TXT 批量 → （可选）语音克隆。"""
    print("开始测试 Spark-TTS FastAPI 服务...")
    t0 = time.time()

    # 1) 健康检查
    hz = healthz()

    # 2) 最小 JSON（虚拟音色）
    synthesize_virtual([
        "今天16:30开会，增长12.5%，报名费￥199。",
        "ID 10086 不要改读法。"
    ])

    # 3) 并行度演示（CPU 可适当尝试 2~3；GPU 通常保持 1）
    synthesize_parallel(["a", "b", "c"], num_workers=3, allow_parallel_on_gpu=None)

    # 4) TXT 批量
    synthesize_txt_upload(TXT_FILE, filename_from_txt=True, sep="|")

    # 5) （可选）语音克隆
    if PROMPT_WAV:
        synthesize_clone(["你好，这是克隆测试。"], prompt_wav=PROMPT_WAV, prompt_text="这里填参考音频的转写")

    print(f"\n全部完成，用时：{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
app_fastapi_tts.py
==================

将 Spark-TTS 封装为一个可 HTTP 调用的 FastAPI 服务。

功能亮点
--------
1) 自动检测 CPU/GPU：
   - 若检测到 CUDA 且有可用 GPU：自动添加 --device 0（GPU 推理）
   - 否则：不传 --device（CPU 推理）

2) 两种合成模式：
   - 语音克隆：传入 prompt_speech_path（可选 prompt_text）
   - 虚拟音色：不传参考音频时，需提供 gender/pitch/speed（枚举值）

3) 批量文本：
   - JSON：texts 列表（每项 { "text": "..." }）
   - TXT：上传 .txt（每行一条；或“文件名|文本”）

4) 并行执行（谨慎开启）：
   - 默认并行度 num_workers=1
   - GPU 上默认禁用并行（更稳），可通过环境变量或请求参数放开
   - CPU 上可适当提高并行度（例如 2~4），实际收益取决于硬件与负载

5) 其他：
   - 根路径 / 自动重定向到 /docs，提供 favicon 占位，控制台更干净
   - 可选简易鉴权（X-API-Key）
   - 返回结构化结果（每条任务的 returncode/stdout/stderr）

运行方式
--------
在 Spark-TTS 仓库根目录执行（建议在已安装好 Spark-TTS 依赖的虚拟环境里）：
    uvicorn app_fastapi_tts:app --host 0.0.0.0 --port 8000 --reload

常用测试
--------
JSON（虚拟音色）：
    curl -X POST "http://127.0.0.1:8000/synthesize" ^
         -H "Content-Type: application/json" ^
         -d "{\"texts\":[{\"text\":\"你好，这是第一条。\"},{\"text\":\"这是第二条。\"}],\"gender\":\"female\",\"pitch\":\"moderate\",\"speed\":\"moderate\"}"

JSON（语音克隆）：
    curl -X POST "http://127.0.0.1:8000/synthesize" ^
         -H "Content-Type: application/json" ^
         -d "{\"texts\":[{\"text\":\"你好，这是克隆测试。\"}],\"prompt_speech_path\":\"assets/ref.wav\",\"prompt_text\":\"这里填参考音频的转写\",\"model_path\":\"D:/GitHub/Spark-TTS/pretrained_models/Spark-TTS-0.5B\"}"

TXT 上传：
    curl -X POST "http://127.0.0.1:8000/synthesize_txt" ^
         -F "file=@D:/path/to/texts.txt" ^
         -F "gender=female" -F "pitch=moderate" -F "speed=moderate" ^
         -F "model_path=D:/GitHub/Spark-TTS/pretrained_models/Spark-TTS-0.5B"

注意事项
--------
- PyTorch（torch）若需 GPU，请确保已安装 CUDA 对应版本；否则即使有显卡也会走 CPU。
- 模型目录（model_path）可共享，无需重复下载（只要路径可读）。
- Windows 下路径推荐使用正斜杠（D:/...），或确保服务进程有权限读取。

"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Body
from fastapi.responses import JSONResponse, RedirectResponse, PlainTextResponse
from pydantic import BaseModel, Field
from concurrent.futures import ProcessPoolExecutor, as_completed

from text_frontend.normalizer import TextNormalizer
import re
import logging

logger = logging.getLogger("sparktts.app")

import re

# -------- 预编译模式（保护区 + 识别数学/自然语言） --------
_RE_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)     # fenced code
_RE_CODE_INLINE = re.compile(r"`[^`]*`")                 # inline code
_RE_MATH_INLINE = re.compile(r"\$(?:\\.|[^$])+\$")       # $...$ latex

# URL / 域名 / 协议
_RE_URL = re.compile(r"\b[a-z]+://[^\s]+", re.IGNORECASE)
_RE_WWW = re.compile(r"\bwww\.[^\s]+", re.IGNORECASE)

# 路径/盘符/相对路径
_RE_WIN_DRIVE = re.compile(r"\b[A-Za-z]:/[^\s]+")
_RE_UNIX_PATH = re.compile(r"(?<!\w)/(?:[^/\s]+/)*[^/\s]+")  # 朴素
_RE_REL_PATH = re.compile(r"\b[\w.\-]+/(?:[\w.\-]+/)*[\w.\-]+")

# 日期
_RE_DATE = re.compile(r"\b\d{4}/\d{1,2}/\d{1,2}\b")

# 正则字面量 /pattern/flags
_RE_REGEX = re.compile(r"(?<!\w)/(?:\\.|[^/])+/[a-z]*", re.IGNORECASE)

# 单位与版本（可选）
_UNITS = {"km/h","m/s","MB/s","GB/s","kb/s","mb/s","gb/s","fps"}
_RE_VER = re.compile(r"\bv\d+/\w+\b", re.IGNORECASE)

# 数学环境：变量或数字两侧的斜杠，含括号表达式
_RE_MATH_TOKEN = r"(?:\d+(?:\.\d+)?|[a-zA-Z]|\([^()]+\))"
_RE_MATH_PAIR = re.compile(rf"({_RE_MATH_TOKEN})\s*/\s*({_RE_MATH_TOKEN})")

# 自然语言并列：两侧为 CJK 或“≥2字符英文词”
_RE_WORD = r"(?:[\u4e00-\u9fff]+|[A-Za-z]{2,})"
_RE_NATURAL_PAIR = re.compile(rf"({_RE_WORD})\s*/\s*({_RE_WORD})")

def _mask_segments(s: str, patterns):
    """用占位符暂存被保护段，返回新串与恢复函数。"""
    slots = []
    def repl(m):
        slots.append(m.group(0))
        return f"@@MASK{len(slots)-1}@@"
    for pat in patterns:
        s = pat.sub(repl, s)
    def unmask(x):
        for i, val in enumerate(slots):
            x = x.replace(f"@@MASK{i}@@", val)
        return x
    return s, unmask

def smart_read_slash(s: str,
                     math_as: str = "除以",
                     natural_as: str = "或",
                     literal_as: str = "",       # 空串=不改
                     protect_units: set = _UNITS,
                     protect_versions: bool = True,
                     protect_regex: bool = True):
    # 1) 构造保护模式列表
    protect_list = [
        _RE_CODE_FENCE, _RE_CODE_INLINE, _RE_MATH_INLINE,
        _RE_URL, _RE_WWW, _RE_WIN_DRIVE, _RE_UNIX_PATH, _RE_REL_PATH,
        _RE_DATE,
    ]
    if protect_versions:
        protect_list.append(_RE_VER)
    if protect_regex:
        protect_list.append(_RE_REGEX)

    # 2) 先整体遮罩保护段
    s_masked, unmask = _mask_segments(s, protect_list)

    # 2.1 单位白名单直接遮罩（逐个替换以免误伤）
    for u in protect_units:
        s_masked = s_masked.replace(u, f"@@UNIT_{u}@@")

    # 3) 在剩余文本上按优先级替换
    # 3.1 数学：A/B -> A 除以 B
    if math_as:
        s_masked = _RE_MATH_PAIR.sub(rf"\1 {math_as} \2", s_masked)

    # 3.2 自然语言：词/词 -> 词 或 词
    if natural_as:
        s_masked = _RE_NATURAL_PAIR.sub(rf"\1 {natural_as} \2", s_masked)

    # 3.3 兜底：把孤立斜杠读“斜杠”（可选）
    if literal_as:
        # 避免把“//”注释类吞掉：先保护双斜杠
        s_masked = s_masked.replace("//", "@@D_SLASH@@")
        s_masked = s_masked.replace("/", literal_as)
        s_masked = s_masked.replace("@@D_SLASH@@", "//")  # 也可还原为“//”

    # 4) 恢复单位与保护段
    for u in protect_units:
        s_masked = s_masked.replace(f"@@UNIT_{u}@@", u)

    return unmask(s_masked)


# ======================== 配置区域（可用环境变量覆盖） ========================

# Spark-TTS 仓库根目录（包含 cli/、webui.py 等）
REPO_ROOT = Path(os.environ.get("SPARK_TTS_REPO_ROOT") or Path(__file__).resolve().parent)

# 模型目录默认指向：pretrained_models/Spark-TTS-0.5B
DEFAULT_MODEL_DIR = Path(
    os.environ.get("SPARK_TTS_MODEL_PATH") or (REPO_ROOT / "pretrained_models" / "Spark-TTS-0.5B")
)

# 默认输出目录
DEFAULT_OUTPUT_DIR = Path(os.environ.get("SPARK_TTS_OUTPUT_DIR") or (REPO_ROOT / "outputs"))
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GPU 上是否允许并行（默认 False 更稳）
ALLOW_PARALLEL_ON_GPU = os.environ.get("SPARK_TTS_ALLOW_PARALLEL_ON_GPU", "0") == "1"

# 可选简易鉴权：设置 API_KEY 后，调用方需在请求头携带 X-API-Key
API_KEY = os.environ.get("API_KEY", "")

# 词库
LEXICON_PATH = REPO_ROOT / "text_frontend" / "lexicon.yaml"
TEXT_NORMALIZER = TextNormalizer(LEXICON_PATH)
# ======================== 工具函数 ========================
def get_normalizer(mode: Optional[str]):
    """
    (mode -> TextNormalizer) 简单缓存，避免反复构造；
    mode 取值：keep/spell/auto/None（None 等同 'auto'）
    """
    from text_frontend.normalizer import TextNormalizer
    import threading

    if not hasattr(get_normalizer, "_cache"):
        get_normalizer._cache = {}   # type: ignore[attr-defined]
        get_normalizer._lock = threading.RLock()  # type: ignore[attr-defined]

    m = mode if mode in {"keep", "spell", "auto"} else "auto"

    with getattr(get_normalizer, "_lock"):  # type: ignore[attr-defined]
        cached = getattr(get_normalizer, "_cache").get(m)  # type: ignore[attr-defined]
        if cached is not None:
            return cached
        n = TextNormalizer(LEXICON_PATH, english_mode=m)
        getattr(get_normalizer, "_cache")[m] = n  # type: ignore[attr-defined]
        return n

def pick_device_arg() -> List[str]:
    """
    自动检测设备：
    - 若检测到 CUDA 且有可用 GPU → 返回 ["--device", "0"]
    - 否则 → 返回 []
    说明：CLI 的 --device 参数类型是 int（GPU id），CPU 模式下不要传该参数。
    """
    try:
        import torch  # 仅用于检测，避免在全局 import 降低冷启动速度
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return ["--device", "0"]
    except Exception as e:
        print(f"[WARN] CUDA 检测失败：{e!r}，将使用 CPU。")
    return []

def _strip_or_extract_pinyin_tags(text: str) -> str:
    # 暂时：去掉 {词|拼音} 保留“词”本体。
    # 若后端升级支持拼音，这里改成解析并传给新接口即可。
    import re
    return re.sub(r"\{([^|{}]+)\|[^{}]+\}", r"\1", text)

def build_cli_argv(
    *,
    text: str,
    model_path: Path,
    output_dir: Path,
    use_clone: bool,
    prompt_speech_path: Optional[Path],
    prompt_text: Optional[str],
    gender: Literal["male", "female"],
    pitch: Literal["very_low", "low", "moderate", "high", "very_high"],
    speed: Literal["very_low", "low", "moderate", "high", "very_high"],
    device_arg: List[str],
) -> List[str]:
    """
    按 Spark-TTS 官方 CLI 规范构造命令行参数（只传必要参数，避免 None/空串）：
    - 注意：CLI 仍使用 --model_dir 参数名（Spark-TTS 的接口定义），
      虽然在服务内部变量名用的是 model_path。
    """
    text = _strip_or_extract_pinyin_tags(text)

    argv = [
        sys.executable, "-m", "cli.inference",
        "--text", text,
        "--save_dir", str(output_dir),
        "--model_dir", str(model_path),
    ]
    if use_clone:
        argv += ["--prompt_speech_path", str(prompt_speech_path)]
        if prompt_text and prompt_text.strip():
            argv += ["--prompt_text", prompt_text.strip()]
    else:
        argv += ["--gender", gender, "--pitch", pitch, "--speed", speed]

    argv += device_arg  # GPU 时追加 ["--device", "0"]；CPU 时为空列表
    return argv


def run_one(argv: List[str], cwd: Path, name: str) -> Dict[str, Any]:
    """
    同步执行单个 CLI 任务，返回结构化结果：
    - returncode：0 表示成功
    - stdout/stderr：子进程的输出，便于排查问题
    - device：gpu/cpu（依据是否传了 --device）
    """
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "name": name,
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "device": "gpu" if ("--device" in argv) else "cpu",
    }


def parse_txt_lines(
    data: str, filename_mode: bool = False, sep: str = "|"
) -> List[Tuple[str, str]]:
    """
    解析 TXT 文本为 [(name, text)] 列表：
    - 默认：每行一条文本，自动生成 name（text_001, text_002, ...）
    - filename_mode=True：支持“文件名|文本”的格式（分隔符可配）
    - 忽略空行与以 # 开头的注释行
    """
    tasks: List[Tuple[str, str]] = []
    for i, raw in enumerate(data.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if filename_mode and sep in line:
            fname, t = line.split(sep, 1)
            fname = fname.strip() or f"text_{i:03d}"
            t = t.strip()
            if t:
                tasks.append((fname, t))
        else:
            tasks.append((f"text_{i:03d}", line))
    return tasks


# ======================== Pydantic 模型（全部中文说明） ========================

class SynthesisItem(BaseModel):
    """单条文本的合成输入"""
    text: str = Field(..., description="要合成的文本")
    # 单条级开关：None 表示继承批量级 normalize_text；True/False 显式覆盖
    normalize: Optional[bool] = Field(default=None, description="是否对该条文本做规范化；未传则继承批量设置")


class SynthesisRequest(BaseModel):
    """
    JSON 请求模型（/synthesize）：
    - texts：至少一条
    - 有参考音色：传 prompt_speech_path（服务器可读），可选 prompt_text
    - 无参考音色：需提供 gender/pitch/speed（枚举）
    - model_path：可选；不传则使用服务默认模型目录
    - output_dir：可选；不传则输出到服务默认 outputs/
    - num_workers：并行度（默认 1；GPU 上默认强制改为 1，除非允许 GPU 并行）
    - english_mode：英文处理模式（keep/spell/auto），默认 auto
    """
    texts: List[SynthesisItem] = Field(default_factory=list, description="要合成的文本列表")

    normalize_text: bool = Field(default=True, description="是否对文本做中文化规范化（全局）；可被单条覆盖")
    english_mode: Literal["keep", "spell", "auto"] = Field(default="auto", description="英文处理模式")

    output_dir: Optional[str] = Field(default=None, description="输出目录（不传则为服务默认 outputs/）")
    model_path: Optional[str] = Field(default=None, description="模型目录（不传则为默认 pretrained_models/Spark-TTS-0.5B）")

    # 语音克隆
    prompt_speech_path: Optional[str] = Field(default=None, description="参考音频路径（服务器可读）")
    prompt_text: Optional[str] = Field(default=None, description="参考音频的转写文本（可选，但建议提供）")

    # 虚拟音色（当没有参考音频时必填）
    gender: Literal["male", "female"] = Field(default="female", description="音色性别")
    pitch: Literal["very_low", "low", "moderate", "high", "very_high"] = Field(default="moderate", description="音高枚举")
    speed: Literal["very_low", "low", "moderate", "high", "very_high"] = Field(default="moderate", description="语速枚举")

    # 调度
    num_workers: int = Field(default=1, ge=1, le=16, description="并行度（默认 1）")
    allow_parallel_on_gpu: Optional[bool] = Field(default=None, description="是否允许在 GPU 上并行（默认遵循服务全局设置）")

    # 仅 /synthesize_txt 用到的解析开关（保持接口对齐；/synthesize 会忽略）
    filename_from_txt: bool = Field(default=False, description="TXT 解析：是否使用“文件名|文本”")
    txt_separator: str = Field(default="|", description="TXT 解析：文件名与文本的分隔符")



class SynthesisResult(BaseModel):
    """单条任务的返回结果"""
    name: str = Field(..., description="任务名（内部标识）")
    device: str = Field(..., description="本次任务使用的设备：gpu/cpu")
    returncode: int = Field(..., description="子进程返回码，0 表示成功")
    stdout: str = Field(..., description="子进程标准输出")
    stderr: str = Field(..., description="子进程标准错误")


class SynthesisResponse(BaseModel):
    """整批任务的返回结果"""
    used_model_path: str = Field(..., description="实际使用的模型目录")
    used_output_dir: str = Field(..., description="实际使用的输出目录")
    used_device: str = Field(..., description="整体判定的设备：gpu/cpu")
    num_workers: int = Field(..., description="实际并行度")
    results: List[SynthesisResult] = Field(..., description="按任务列出的结果明细")


# ======================== FastAPI 应用与路由 ========================

app = FastAPI(
    title="Spark-TTS HTTP 服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def _check_api_key(x_api_key: Optional[str]) -> None:
    """
    可选的简单鉴权：
    - 若未设置 API_KEY 环境变量，则不校验。
    - 若设置了 API_KEY，则要求请求头带 X-API-Key 且与之匹配。
    """
    if not API_KEY:
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="未授权（X-API-Key 无效或缺失）")


@app.get("/", include_in_schema=False)
def root():
    """根路径重定向到 Swagger 文档"""
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """避免浏览器请求 favicon 时出现 404 噪音日志"""
    return PlainTextResponse("", status_code=200)


@app.get("/healthz")
def healthz():
    lex_path = str(LEXICON_PATH)
    try:
        mtime = os.path.getmtime(LEXICON_PATH)
    except Exception:
        mtime = None
    return {
        "status": "ok",
        "repo_root": str(REPO_ROOT),
        "default_model_path": str(DEFAULT_MODEL_DIR),
        "default_output_dir": str(DEFAULT_OUTPUT_DIR),
        "allow_parallel_on_gpu": ALLOW_PARALLEL_ON_GPU,
        # 新增：词典状态
        "lexicon_path": lex_path,
        "lexicon_mtime": mtime,
    }


@app.post("/text/normalize_preview")
def normalize_preview(
    body: Dict[str, Any] = Body(
        ...,
        example={
            "texts": ["iPhone 15 Pro 将在 2025年8月21日 16:30 发布。", "ID 10086 不要改读法。"],
            "normalize": True,
            "english_mode": "auto"  # keep / spell / auto
        },
        description="调试文本规范化：支持 english_mode"
    ),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_api_key(x_api_key)

    texts: List[str] = body.get("texts", [])
    if not isinstance(texts, list):
        raise HTTPException(status_code=400, detail="texts 必须是字符串列表")

    normalize_flag: bool = bool(body.get("normalize", True))
    english_mode: Optional[str] = body.get("english_mode", "auto")

    norm = get_normalizer(english_mode)
    english_mode_used = getattr(norm, "english_mode", "auto")

    items = []
    for t in texts:
        s = str(t)
        normed = norm.normalize(s) if normalize_flag else s

        # 扫描词典命中，帮助判断英文是否被强制改读
        hits_keep, hits_force = [], []
        try:
            lex = norm.lex
            src = s
            hits_keep = [k for k in (lex.keep_english or []) if k and k in src]
            hits_force = [k for k in (lex.force_zh or {}).keys() if k and k in src]
        except Exception:
            pass

        items.append({
            "original": s,
            "normalized": normed,
            "lexicon_hits": {"keep_english": hits_keep, "force_zh": hits_force}
        })

    try:
        logger.info("normalize_preview count=%d mode=%s", len(items), english_mode_used)
    except Exception:
        pass

    return {
        "normalized": [it["normalized"] for it in items],
        "english_mode_used": english_mode_used,
        "items": items
    }


@app.post("/lexicon/reload")
def lexicon_reload(x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    from text_frontend.normalizer import TextNormalizer

    global TEXT_NORMALIZER
    TEXT_NORMALIZER = TextNormalizer(LEXICON_PATH, english_mode="auto")

    # 清理 get_normalizer 缓存
    try:
        if hasattr(get_normalizer, "_cache"):
            get_normalizer._cache.clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        mtime = os.path.getmtime(LEXICON_PATH)
    except Exception:
        mtime = None
    return {"status": "reloaded", "lexicon_path": str(LEXICON_PATH), "lexicon_mtime": mtime}


@app.on_event("startup")
def app_startup_warmup():
    """进程启动预热，降低首请求时延。"""
    try:
        for m in ("auto", "keep", "spell"):
            n = get_normalizer(m)
            _ = n.normalize("预热 123，在 2025年8月21日 16:30 发布。ID 10086。")
        logger.info("Spark-TTS normalizer warmup done.")
    except Exception as e:
        logger.warning("Spark-TTS normalizer warmup skipped: %r", e)



@app.post("/synthesize", response_model=SynthesisResponse)
def synthesize(req: SynthesisRequest, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)

    if not req.texts:
        raise HTTPException(status_code=400, detail="texts 不能为空")

    # 解析模型与输出路径
    model_p = Path(req.model_path) if req.model_path else DEFAULT_MODEL_DIR
    if not model_p.exists():
        raise HTTPException(status_code=404, detail=f"模型目录不存在：{model_p}")

    output_p = Path(req.output_dir) if req.output_dir else DEFAULT_OUTPUT_DIR
    output_p.mkdir(parents=True, exist_ok=True)

    # 模式判定：有无参考音频
    use_clone = req.prompt_speech_path is not None
    prompt_wav = Path(req.prompt_speech_path) if use_clone else None
    if use_clone and not prompt_wav.is_file():
        raise HTTPException(status_code=404, detail=f"参考音频不存在：{prompt_wav}")

    # 设备检测
    device_arg = pick_device_arg()
    gpu_in_use = bool(device_arg)

    # 并行策略
    allow_parallel = req.allow_parallel_on_gpu if req.allow_parallel_on_gpu is not None else ALLOW_PARALLEL_ON_GPU
    num_workers = req.num_workers
    if gpu_in_use and not allow_parallel and num_workers > 1:
        num_workers = 1

    # 选定 normalizer（按 english_mode 缓存复用）
    norm = get_normalizer(req.english_mode)

    # 组装任务
    jobs: List[Tuple[List[str], Path, str]] = []
    for i, item in enumerate(req.texts, 1):
        name = f"text_{i:03d}"
        use_norm = item.normalize if item.normalize is not None else req.normalize_text
        text = norm.normalize(item.text) if use_norm else item.text
        text = smart_read_slash(text)  # 对于 / 的不同情景发音的设置
        argv = build_cli_argv(
            text=text,
            model_path=model_p,
            output_dir=output_p,
            use_clone=use_clone,
            prompt_speech_path=prompt_wav,
            prompt_text=req.prompt_text,
            gender=req.gender,
            pitch=req.pitch,
            speed=req.speed,
            device_arg=device_arg,
        )
        jobs.append((argv, REPO_ROOT, name))

    # 执行任务
    results: List[Dict[str, Any]] = []
    if num_workers <= 1:
        for argv, cwd, name in jobs:
            results.append(run_one(argv, cwd, name))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            fut_map = {ex.submit(run_one, argv, cwd, name): (argv, name) for argv, cwd, name in jobs}
            for fut in as_completed(fut_map):
                try:
                    results.append(fut.result())
                except Exception as e:
                    argv, name = fut_map[fut]
                    results.append({
                        "name": name,
                        "device": "gpu" if gpu_in_use else "cpu",
                        "returncode": -999,
                        "stdout": "",
                        "stderr": f"并行子进程异常：{repr(e)}",
                    })

    return SynthesisResponse(
        used_model_path=str(model_p),
        used_output_dir=str(output_p),
        used_device="gpu" if gpu_in_use else "cpu",
        num_workers=num_workers,
        results=[SynthesisResult(**{k: v for k, v in r.items() if k in SynthesisResult.__fields__}) for r in results],
    )



@app.post("/synthesize_txt", response_model=SynthesisResponse)
async def synthesize_txt(
    file: UploadFile = File(..., description="UTF-8 文本文件；每行一条；或 '文件名|文本'"),
    filename_from_txt: bool = Form(default=False),
    txt_separator: str = Form(default="|"),
    output_dir: Optional[str] = Form(default=None),
    model_path: Optional[str] = Form(default=None),
    # 参考音色
    prompt_speech_path: Optional[str] = Form(default=None),
    prompt_text: Optional[str] = Form(default=None),
    # 虚拟音色（无参考音色时必填）
    gender: Literal["male", "female"] = Form(default="female"),
    pitch: Literal["very_low", "low", "moderate", "high", "very_high"] = Form(default="moderate"),
    speed: Literal["very_low", "low", "moderate", "high", "very_high"] = Form(default="moderate"),
    # 并行
    num_workers: int = Form(default=1),
    allow_parallel_on_gpu: Optional[bool] = Form(default=None),
    # 规范化
    normalize_text: bool = Form(default=True),
    english_mode: Literal["keep", "spell", "auto"] = Form(default="auto"),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_api_key(x_api_key)

    content = (await file.read()).decode("utf-8", errors="replace")
    tasks = parse_txt_lines(content, filename_mode=filename_from_txt, sep=txt_separator)
    if not tasks:
        raise HTTPException(status_code=400, detail="txt 文件内容为空")

    model_p = Path(model_path) if model_path else DEFAULT_MODEL_DIR
    if not model_p.exists():
        raise HTTPException(status_code=404, detail=f"模型目录不存在：{model_p}")

    output_p = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_p.mkdir(parents=True, exist_ok=True)

    use_clone = prompt_speech_path is not None
    prompt_wav = Path(prompt_speech_path) if use_clone else None
    if use_clone and not prompt_wav.is_file():
        raise HTTPException(status_code=404, detail=f"参考音频不存在：{prompt_wav}")

    device_arg = pick_device_arg()
    gpu_in_use = bool(device_arg)

    allow_parallel = allow_parallel_on_gpu if allow_parallel_on_gpu is not None else ALLOW_PARALLEL_ON_GPU
    if gpu_in_use and not allow_parallel and num_workers > 1:
        num_workers = 1

    # 选定 normalizer（按 english_mode 缓存复用）
    norm = get_normalizer(english_mode)

    jobs: List[Tuple[List[str], Path, str]] = []
    for name, text in tasks:
        txt = norm.normalize(text) if normalize_text else text
        txt = smart_read_slash(txt)  # 对于 / 的不同情景发音的设置
        argv = build_cli_argv(
            text=txt,
            model_path=model_p,
            output_dir=output_p,
            use_clone=use_clone,
            prompt_speech_path=prompt_wav,
            prompt_text=prompt_text,
            gender=gender,
            pitch=pitch,
            speed=speed,
            device_arg=device_arg,
        )
        jobs.append((argv, REPO_ROOT, name))

    results: List[Dict[str, Any]] = []
    if num_workers <= 1:
        for argv, cwd, name in jobs:
            results.append(run_one(argv, cwd, name))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            fut_map = {ex.submit(run_one, argv, cwd, name): (argv, name) for argv, cwd, name in jobs}
            for fut in as_completed(fut_map):
                try:
                    results.append(fut.result())
                except Exception as e:
                    argv, name = fut_map[fut]
                    results.append({
                        "name": name,
                        "device": "gpu" if gpu_in_use else "cpu",
                        "returncode": -999,
                        "stdout": "",
                        "stderr": f"并行子进程异常：{repr(e)}",
                    })

    return SynthesisResponse(
        used_model_path=str(model_p),
        used_output_dir=str(output_p),
        used_device="gpu" if gpu_in_use else "cpu",
        num_workers=num_workers,
        results=[SynthesisResult(**{k: v for k, v in r.items() if k in SynthesisResult.__fields__}) for r in results],
    )

if __name__ == '__main__':
    bash = 'uvicorn app_fastapi_tts:app --host 0.0.0.0 --port 8001 --reload'
    """ 关闭 8000、8001 端口，有时候不知道为什么一直占用，现阶段先不研究这么细了
for /f "tokens=5" %p in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /PID %p >nul 2>&1
for /f "tokens=5" %p in ('netstat -ano ^| findstr ":8001" ^| findstr "LISTENING"') do taskkill /F /PID %p >nul 2>&1
    """
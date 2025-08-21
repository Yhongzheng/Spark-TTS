# -*- coding: utf-8 -*-
"""
Spark-TTS 批量合成封装
=====================

功能特性：
- 自动检测 CPU/GPU：若安装了 CUDA 版 PyTorch 且有可用 GPU，则使用 GPU(默认 id=0)；否则使用 CPU。
- 支持两种模式：
  1) 语音克隆（提供参考音频 prompt_speech_path，可选 prompt_text）
  2) 虚拟音色（不提供参考音频时必须给 gender/pitch/speed）
- 批量输入：
  - 直接传入文本列表 texts
  - 或从 TXT 文件加载（默认每行一条文本；也支持 “文件名|文本” 的格式）
- 默认输出目录为 ./outputs/（可通过参数修改）
- 并行执行（ProcessPoolExecutor + 子进程调用 `python -m cli.inference`），避免 GIL 与相互干扰。
  - **重要提示**：TTS 推理很吃算力。GPU 环境同时跑多个任务可能抢占显存、相互拖慢；CPU 环境并行度也需保守。
  - 默认并行度为 1；你可以传入 num_workers>1 来尝试并行。
- 健壮的错误提示、参数校验、详细返回结果。

使用建议：
- **GPU 环境**：建议 num_workers=1（除非你非常确定显存足够且愿意承担速度不可预测的代价）。
- **CPU 环境**：可适度提高并行度（例如 2~4），但总时长不一定线性下降，视你的 CPU 内核与负载而定。
- **长文本**：官方 CLI 不一定为长段落做特殊切分；如果要处理超长文本，建议自行预切段后批量合成再拼接。

作者注：脚本默认在“仓库根目录”作为工作目录运行；如果改到其他位置使用，请把 repo_root 设为 Spark-TTS 仓库根。
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed


class SparkTTSBatchSynthesizer:
    """
    Spark-TTS 批量合成器

    参数
    ----
    model_dir : str | Path
        预训练模型目录（例如：pretrained_models/Spark-TTS-0.5B）
    output_dir : str | Path, optional
        输出目录，默认 ./outputs/
    repo_root : str | Path, optional
        Spark-TTS 仓库根目录（包含 cli/、webui.py 等）。默认取当前脚本所在目录。
    prefer_gpu : bool
        是否偏好使用 GPU（若可用）。默认 True。
    gpu_id : int
        使用的 GPU id（当 GPU 可用且 prefer_gpu=True 时生效）。默认 0。
    allow_parallel_on_gpu : bool
        GPU 可用时是否允许并行。默认 False（更稳）。
    """

    # pitch/speed 的合法取值（根据 CLI help）
    VALID_PITCH = {"very_low", "low", "moderate", "high", "very_high"}
    VALID_SPEED = {"very_low", "low", "moderate", "high", "very_high"}
    VALID_GENDER = {"male", "female"}

    def __init__(
        self,
        model_dir: Path | str,
        output_dir: Path | str | None = None,
        repo_root: Path | str | None = None,
        prefer_gpu: bool = True,
        gpu_id: int = 0,
        allow_parallel_on_gpu: bool = False,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir) if output_dir else (self.repo_root / "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.prefer_gpu = prefer_gpu
        self.gpu_id = int(gpu_id)
        self.allow_parallel_on_gpu = allow_parallel_on_gpu

        # 基础校验
        if not (self.repo_root / "cli").is_dir():
            print("【警告】未检测到 repo_root 下的 cli/ 目录，请确认 repo_root 是否为 Spark-TTS 仓库根目录。")
        if not self.model_dir.exists():
            raise FileNotFoundError(f"未找到模型目录：{self.model_dir}")

    # ---------------- 公共主入口 ----------------

    def synthesize(
        self,
        texts: Optional[Iterable[str]] = None,
        text_file: Optional[Path | str] = None,
        *,
        # 语音克隆相关（两者至少给 prompt_speech_path，其中文本可选）
        prompt_speech_path: Optional[Path | str] = None,
        prompt_text: Optional[str] = None,
        # 虚拟音色相关（当没有参考音频时必须提供）
        gender: str = "female",
        pitch: str = "moderate",
        speed: str = "moderate",
        # 并行/调度
        num_workers: int = 1,
        filename_from_txt: bool = False,
        txt_separator: str = "|",
        encoding: str = "utf-8",
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        批量合成语音的高层封装。

        传参用法：
        - texts：直接传入字符串可迭代对象，每个元素是一条要合成的文本。
        - text_file：从 TXT 文件读取文本（默认每行一条；若 filename_from_txt=True，支持“文件名|文本”格式）。
        - prompt_speech_path + 可选 prompt_text → 语音克隆模式。
        - 否则必须提供 gender/pitch/speed → 虚拟音色模式。
        - output_dir 未传则默认 ./outputs/。
        - 并行执行：num_workers>1 时使用进程池并行。

        返回：
        - 列表，元素为字典，包含每条文本的状态（returncode、stdout/stderr、使用设备、文本内容等）。
        """
        # 收集任务
        tasks = self._prepare_tasks(
            texts=texts,
            text_file=text_file,
            filename_from_txt=filename_from_txt,
            txt_separator=txt_separator,
            encoding=encoding,
        )
        if not tasks:
            raise ValueError("没有可合成的文本：texts 为空且 text_file 未提供或内容为空。")

        # 参数校验与模式确认
        use_clone = prompt_speech_path is not None
        if use_clone:
            prompt_speech_path = Path(prompt_speech_path)
            if not prompt_speech_path.is_file():
                raise FileNotFoundError(f"参考音频不存在：{prompt_speech_path}")
        else:
            # 无参考音频 → 必须提供虚拟音色控制参数
            if gender not in self.VALID_GENDER:
                raise ValueError(f"gender 必须为 {self.VALID_GENDER}，当前：{gender!r}")
            if pitch not in self.VALID_PITCH:
                raise ValueError(f"pitch 必须为 {self.VALID_PITCH}，当前：{pitch!r}")
            if speed not in self.VALID_SPEED:
                raise ValueError(f"speed 必须为 {self.VALID_SPEED}，当前：{speed!r}")

        # 设备检测
        device_arg = self._pick_device_arg()
        gpu_in_use = bool(device_arg)

        # 并行度策略提示
        if gpu_in_use and not self.allow_parallel_on_gpu and num_workers > 1:
            print("【提示】检测到 GPU 可用，但并行在 GPU 上可能导致显存争用与减速。"
                  "已将并行度强制设为 1。若你确实要在 GPU 上并行，请在实例化时设置 allow_parallel_on_gpu=True。")
            num_workers = 1

        # 构建每个任务的命令参数
        jobs: List[Tuple[List[str], Path, str]] = []
        for (name, text) in tasks:
            argv = self._build_cli_argv(
                text=text,
                use_clone=use_clone,
                prompt_speech_path=prompt_speech_path,
                prompt_text=prompt_text,
                gender=gender,
                pitch=pitch,
                speed=speed,
                device_arg=device_arg,
            )
            jobs.append((argv, self.repo_root, name))

        # 执行
        if dry_run:
            for argv, _, name in jobs:
                print(f"[DRY-RUN] 任务: {name} -> argv = {argv}")
            return []

        results: List[Dict[str, Any]] = []
        if num_workers <= 1:
            for argv, cwd, name in jobs:
                res = self._run_one(argv, cwd, name)
                results.append(res)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                fut_map = {ex.submit(self._subprocess_runner, argv, cwd, name): (argv, name)
                           for argv, cwd, name in jobs}
                for fut in as_completed(fut_map):
                    argv, name = fut_map[fut]
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        results.append({
                            "name": name,
                            "argv": argv,
                            "returncode": -999,
                            "stdout": "",
                            "stderr": f"并行子进程异常：{repr(e)}",
                            "device": "gpu" if gpu_in_use else "cpu",
                        })

        return results

    # ---------------- 工具方法 ----------------

    def _prepare_tasks(
        self,
        texts: Optional[Iterable[str]],
        text_file: Optional[str | Path],
        *,
        filename_from_txt: bool,
        txt_separator: str,
        encoding: str,
    ) -> List[Tuple[str, str]]:
        """
        生成 (name, text) 列表。
        - name 仅用于标识（日志/返回结果中展示），不一定等于最终文件名（CLI 内部有自己的命名策略）。
        - 支持：
          1) 直接从 texts 收集（自动给 name 编号）
          2) 从 txt 文件读取：
             - 默认每行一条文本，空行或以 # 开头的行会被跳过
             - 若 filename_from_txt=True，则支持 “文件名|文本” 形式（分隔符可配）
        """
        tasks: List[Tuple[str, str]] = []

        if texts:
            for i, t in enumerate(texts, 1):
                tt = (t or "").strip()
                if tt:
                    tasks.append((f"text_{i:03d}", tt))

        if text_file:
            p = Path(text_file)
            if not p.is_file():
                raise FileNotFoundError(f"TXT 文件不存在：{p}")
            with p.open("r", encoding=encoding) as f:
                for i, line in enumerate(f, 1):
                    raw = line.strip()
                    if not raw or raw.startswith("#"):
                        continue
                    if filename_from_txt and txt_separator in raw:
                        fname, txt = raw.split(txt_separator, 1)
                        fname = fname.strip()
                        txt = txt.strip()
                        if txt:
                            tasks.append((fname or f"text_{i:03d}", txt))
                    else:
                        tasks.append((f"text_{i:03d}", raw))
        return tasks

    def _pick_device_arg(self) -> List[str]:
        """
        自动检测设备：
        - 若 prefer_gpu 且 CUDA 可用 → ["--device", str(gpu_id)]
        - 否则 → []
        """
        if not self.prefer_gpu:
            return []
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return ["--device", str(self.gpu_id)]
        except Exception as e:
            print(f"【警告】CUDA 检测失败：{e!r}，将使用 CPU。")
        return []

    def _build_cli_argv(
        self,
        *,
        text: str,
        use_clone: bool,
        prompt_speech_path: Optional[Path],
        prompt_text: Optional[str],
        gender: str,
        pitch: str,
        speed: str,
        device_arg: List[str],
    ) -> List[str]:
        """
        按 CLI 规范拼装命令行参数（只传必要参数，避免 None/空串误传）。
        """
        argv = [
            sys.executable, "-m", "cli.inference",
            "--text", text,
            "--save_dir", str(self.output_dir),
            "--model_dir", str(self.model_dir),
        ]
        if use_clone:
            argv += ["--prompt_speech_path", str(prompt_speech_path)]
            if prompt_text and prompt_text.strip():
                argv += ["--prompt_text", prompt_text.strip()]
        else:
            argv += ["--gender", gender, "--pitch", pitch, "--speed", speed]

        argv += device_arg  # GPU 时添加 --device <id>；CPU 时为空
        return argv

    @staticmethod
    def _subprocess_runner(argv: List[str], cwd: Path, name: str) -> Dict[str, Any]:
        """
        供进程池调用的静态方法（必须可 picklable）。
        """
        # 注意：不要在子进程里导入 torch 等重型库，以减少启动开销。
        proc = subprocess.run(
            argv, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        return {
            "name": name,
            "argv": argv,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "device": "gpu" if ("--device" in argv) else "cpu",
        }

    def _run_one(self, argv: List[str], cwd: Path, name: str) -> Dict[str, Any]:
        """
        单任务执行（同步）。
        """
        print(f"[INFO] 开始任务：{name}")
        res = self._subprocess_runner(argv, cwd, name)
        if res["returncode"] == 0:
            print(f"[OK] 任务完成：{name}")
        else:
            print(f"[ERR] 任务失败：{name}，返回码 {res['returncode']}。\n----stderr----\n{res['stderr']}\n--------------")
        return res


# ---------------- 下面是一个可直接右键运行的示例 ----------------

def demo() -> None:
    """
    演示：从两条文本合成音频（无参考音频 → 虚拟音色）。
    你也可以改成从 TXT 批量读取（见下面注释）。
    """
    repo_root = Path(__file__).resolve().parent
    model_dir = repo_root / "pretrained_models" / "Spark-TTS-0.5B"

    tts = SparkTTSBatchSynthesizer(
        model_dir=model_dir,
        output_dir=repo_root / "outputs",   # 不填则默认 ./outputs/
        repo_root=repo_root,
        prefer_gpu=True,                    # 自动检测 GPU
        gpu_id=0,
        allow_parallel_on_gpu=False,        # GPU 上默认不并行，更稳
    )

    # 方式 A：直接传列表（虚拟音色）
    texts = [
        "你好，这是第一条测试语音。",
        "这里是第二条测试语音，我们用同样的参数合成。",
    ]
    results = tts.synthesize(
        texts=texts,
        # 如果你要语音克隆，请把 prompt_speech_path 指向有效音频，并注释掉 gender/pitch/speed：
        # prompt_speech_path=repo_root / "assets" / "ref.wav",
        # prompt_text="在这里填写参考音频的转写文本",
        gender="female",
        pitch="moderate",
        speed="moderate",
        num_workers=1,  # CPU 可尝试 >1；GPU 建议 1
    )
    print("汇总结果：", {r["name"]: r["returncode"] for r in results})

    # 方式 B：从 TXT 读取
    # 假设 txt 文件每行一个文本：
    #   repo_root / "assets" / "batch.txt"
    # 或者启用“文件名|文本”格式（filename_from_txt=True），例如：
    #   welcome_001|欢迎使用我们的产品！
    #   tips_002|接下来为您介绍三条使用小贴士……
    #
    # results2 = tts.synthesize(
    #     text_file=repo_root / "assets" / "batch.txt",
    #     filename_from_txt=True,
    #     txt_separator="|",
    #     # 语音克隆示例：
    #     # prompt_speech_path=repo_root / "assets" / "ref.wav",
    #     # prompt_text="在这里填写参考音频的转写文本",
    #     # 虚拟音色示例：
    #     gender="female", pitch="moderate", speed="moderate",
    #     num_workers=1,
    # )
    # print("TXT 结果：", {r["name"]: r["returncode"] for r in results2})


if __name__ == "__main__":
    demo()

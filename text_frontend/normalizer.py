# -*- coding: utf-8 -*-
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Literal

import yaml
import cn2an

from .rules import compile_rules

_DIGITS = "零一二三四五六七八九"


def digits_spell_out(num_str: str) -> str:
    # 逐字读（手机号/长编号）
    return "".join(_DIGITS[int(c)] if c.isdigit() else c for c in num_str)


def an2cn_number(num_str: str) -> str:
    # cn2an 把阿拉伯数字转中文读法（含小数）
    try:
        return cn2an.an2cn(num_str, "low")
    except Exception:
        # 兜底：逐字
        return digits_spell_out(num_str)


@dataclass
class Lexicon:
    keep_english: List[str]
    force_zh: Dict[str, str]
    pinyin_override: Dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "Lexicon":
        if not path.is_file():
            return cls([], {}, {})
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            keep_english=data.get("keep_english", []) or [],
            force_zh=data.get("force_zh", {}) or {},
            pinyin_override=data.get("pinyin_override", {}) or {},
        )


class TextNormalizer:
    """
    english_mode:
      - keep  : 英文原样保留（默认只在 force_zh 命中时改读）
      - spell : 英文转“可拼读”形式（字母以空格分开；数字逐字中文；常见分隔符转中文词）
      - auto  : 启发式选择 keep 或 spell：
                * 命中 keep_english → keep
                * 命中 force_zh → 在恢复阶段强制中文
                * 全大写且 2~6 长度的缩写 → spell（如 CPU、AI、NLP）
                * 其余 → keep
    """
    SEP_MAP = {
        ".": "点",
        "-": "横杠",
        "_": "下划线",
        "/": "斜杠",
        "\\": "反斜杠",
        "+": "加号",
        "#": "井号",
        "@": "艾特",
        "&": "和",
    }

    def __init__(self, lexicon_path: Path, english_mode: Literal["keep", "spell", "auto"] = "auto"):
        self.lex = Lexicon.load(lexicon_path)
        self.rules = compile_rules()
        self.english_mode: Literal["keep", "spell", "auto"] = english_mode
        # 运行期桶：每次 normalize 调用时重置
        self._bucket: List[str] = []

    # -------- 英文 token 处理 --------
    @staticmethod
    def _is_acronym(token: str) -> bool:
        # 纯大写、长度 2~6 认为是缩写
        return 2 <= len(token) <= 6 and token.isupper() and token.isalpha()

    def _spell_token(self, token: str) -> str:
        # 把英文 token 转成“可读”的形式：
        # - 字母：大写后用空格分开（C P U）
        # - 数字：逐字中文（123 → 一 二 三）
        # - 分隔符：转中文词（. → 点，- → 横杠 等）
        out: List[str] = []
        for ch in token:
            if ch.isalpha():
                out.append(ch.upper())
            elif ch.isdigit():
                out.append(_DIGITS[int(ch)])
            elif ch in self.SEP_MAP:
                out.append(self.SEP_MAP[ch])
            else:
                # 其他符号直接加入（必要时也可映射）
                out.append(ch)
        # 连续空格规整
        s = " ".join(x for x in out if x != "")
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s

    def _transform_english(self, token: str) -> str:
        # 1) 词典强制覆盖优先（允许对英文 token 覆盖中文读法）
        if token in self.lex.force_zh:
            return self.lex.force_zh[token]
        # 2) 白名单保留
        if token in self.lex.keep_english:
            return token
        # 3) 按模式处理
        mode = self.english_mode
        if mode == "keep":
            return token
        if mode == "spell":
            return self._spell_token(token)
        # auto
        if self._is_acronym(token):
            return self._spell_token(token)
        # 包含字母+数字但主体是大写字母也倾向拼读，如 "RTX4090"
        letters = ''.join(ch for ch in token if ch.isalpha())
        if letters.isupper() and len(letters) >= 2:
            return self._spell_token(token)
        # 其余保留
        return token

    # -------- 保护/恢复机制 --------
    def _stash(self, token: str) -> str:
        idx = len(self._bucket)
        self._bucket.append(token)
        return f"[[TK{idx}]]"

    def _protect_re(self, m: re.Match) -> str:
        return self._stash(m.group(0))

    def _protect_tokens(self, s: str) -> str:
        # 每次调用重置桶
        self._bucket = []
        # 1) 先保护 keep_english（完全保留）
        for kw in sorted(self.lex.keep_english, key=len, reverse=True):
            if kw:
                s = s.replace(kw, self._stash(kw))
        # 2) 再保护所有包含字母的 token（英专名、缩写、文件名等）
        s = re.sub(r"(?<!\[\[)[A-Za-z][A-Za-z0-9\-\._/\\#@&+]*", self._protect_re, s)
        return s

    def _unprotect(self, s: str) -> str:
        def _restore(m: re.Match) -> str:
            idx = int(m.group(1))
            raw = self._bucket[idx]
            return self._transform_english(raw)

        return re.sub(r"\[\[TK([0-9零一二三四五六七八九]+)]]", _restore, s)

    # -------- 词典中文覆盖（非英文 token）--------
    def _apply_force_zh(self, s: str) -> str:
        # 对整体文本做一次覆盖（适合中文/符号串等）；
        # 对英文 token 的覆盖在 _transform_english 中处理，确保优先级最高。
        for k, v in self.lex.force_zh.items():
            if k:
                s = s.replace(k, v)
        return s

    def _apply_pinyin_override(self, s: str) -> str:
        # 约定：把命中的词替换为 {原词|拼音} 这种“可解析标签”
        # 例如：建模 -> {建模|jian4 mo2}
        # 若下游不识别拼音标签，兜底：退回到 force_zh 同义改写（若配置了）。
        for k, py in self.lex.pinyin_override.items():
            if not k:
                continue
            tagged = f"{{{k}|{py}}}"
            if k in s:
                s = s.replace(k, tagged)
        return s

    # -------- 主流程 --------
    def set_english_mode(self, mode: Literal["keep", "spell", "auto"]) -> None:
        self.english_mode = mode

    def normalize(self, text: str) -> str:
        s = text

        # 0) 保护英文/白名单；再做一次覆盖（中文/符号串）
        s = self._protect_tokens(s)
        s = self._apply_force_zh(s)
        s = self._apply_pinyin_override(s)

        # 1) 年月日 & 时间
        s = self.rules["year"].sub(lambda m: digits_spell_out(m.group(1)), s)  # 年份逐字：二零二五
        s = self.rules["month"].sub(lambda m: an2cn_number(m.group(1)), s)
        s = self.rules["day"].sub(lambda m: an2cn_number(m.group(1)), s)
        s = self.rules["hhmmss"].sub(
            lambda m: f"{an2cn_number(m.group(1))}点{an2cn_number(m.group(2))}分" + (
                f"{an2cn_number(m.group(3))}秒" if m.group(3) else ""),
            s
        )

        # 2) 百分比、区间、序数、货币
        s = self.rules["percent"].sub(lambda m: "百分之" + an2cn_number(m.group(1)), s)
        s = self.rules["range"].sub(lambda m: f"{an2cn_number(m.group(1))}到{an2cn_number(m.group(2))}", s)
        s = self.rules["ordinal"].sub(lambda m: "第" + an2cn_number(m.group(1)), s)
        s = self.rules["currency"].sub(lambda m: "人民币" + an2cn_number(m.group(1)) + "元", s)

        # 3) 手机号/长编号逐字
        s = self.rules["phone11"].sub(lambda m: digits_spell_out(m.group(1)), s)
        s = self.rules["longnum"].sub(lambda m: digits_spell_out(m.group(1)), s)

        # 4) 普通纯数字/小数（不含字母 token）
        s = self.rules["purenum"].sub(lambda m: an2cn_number(m.group(1)), s)

        # 5) 去重“零”、微清洗
        s = re.sub("零+", "零", s).rstrip("零")

        # 6) 恢复被保护的英文专名（此处应用 english_mode / force_zh / keep_english）
        s = self._unprotect(s)
        return s

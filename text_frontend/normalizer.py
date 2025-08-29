# -*- coding: utf-8 -*-
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    def __init__(self, lexicon_path: Path):
        self.lex = Lexicon.load(lexicon_path)
        self.rules = compile_rules()

    def _protect_tokens(self, s: str) -> str:
        # 1) 先保护词典中的 keep_english
        self._bucket = []
        for kw in sorted(self.lex.keep_english, key=len, reverse=True):
            s = s.replace(kw, self._stash(kw))
        # 2) 再保护所有包含字母的 token（英专名、缩写等）
        s = re.sub(r"[A-Za-z][A-Za-z0-9\-\._]*", self._protect_re, s)
        return s

    def _stash(self, token: str) -> str:
        idx = len(self._bucket)
        self._bucket.append(token)
        return f"[[TK{idx}]]"

    def _protect_re(self, m: re.Match) -> str:
        return self._stash(m.group(0))

    def _unprotect(self, s: str) -> str:
        def _restore(m: re.Match) -> str:
            idx = int(m.group(1))
            return self._bucket[idx]
        return re.sub(r"\[\[TK(\d+)]]", _restore, s)

    def _apply_force_zh(self, s: str) -> str:
        # 词典覆盖（最高优先级）：把指定词直接替换为给定中文读法
        for k, v in self.lex.force_zh.items():
            s = s.replace(k, v)
        return s

    def normalize(self, text: str) -> str:
        s = text

        # 0) 词典保护/覆盖
        s = self._protect_tokens(s)
        s = self._apply_force_zh(s)

        # 1) 年月日 & 时间
        s = self.rules["year"].sub(lambda m: digits_spell_out(m.group(1)), s)  # 年份逐字：二零二五
        s = self.rules["month"].sub(lambda m: an2cn_number(m.group(1)), s)
        s = self.rules["day"].sub(lambda m: an2cn_number(m.group(1)), s)
        s = self.rules["hhmmss"].sub(
            lambda m: f"{an2cn_number(m.group(1))}点{an2cn_number(m.group(2))}分" + (f"{an2cn_number(m.group(3))}秒" if m.group(3) else ""),
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

        # 6) 恢复被保护的英文专名
        s = self._unprotect(s)
        return s

# 可选：如果后续想接 WeTextProcessing 做更强的 TN，可以在 normalize() 前后挂钩
# try:
#     from WeTextProcessing import TextNormalizer as WTPNormalizer
#     self.wtp = WTPNormalizer()
# except Exception:
#     self.wtp = None
# if self.wtp: s = self.wtp.normalize(s)

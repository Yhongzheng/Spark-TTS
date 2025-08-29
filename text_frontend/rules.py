# -*- coding: utf-8 -*-
import re
from typing import Callable

# 常用正则：只覆盖“明确且小”的场景，不搞“大而全”
RE_PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")
RE_RANGE   = re.compile(r"(\d+(?:\.\d+)?)\s*[-–—~至]\s*(\d+(?:\.\d+)?)")
RE_CURRENCY= re.compile(r"(?:￥|¥)\s*(\d+(?:\.\d+)?)")
RE_ORDINAL = re.compile(r"第(\d+)(?=[\u4e00-\u9fa5A-Za-z0-9])")  # 第3章/第2期
RE_HHMMSS  = re.compile(r"(?<!\d)(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?")
RE_YEARN   = re.compile(r"(?<!\d)(\d{2,4})(?=年)")
RE_MONTH   = re.compile(r"(?<!\d)(\d{1,2})(?=月)")
RE_DAY     = re.compile(r"(?<!\d)(\d{1,2})(?=日|号)")
RE_LONGNUM = re.compile(r"(?<!\d)(\d{8,})(?!\d)")       # ≥8位编号逐字
RE_PHONE11 = re.compile(r"(?<!\d)(1\d{10})(?!\d)")      # 11位手机号
RE_PURENUM = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)")# 普通纯数字/小数

def compile_rules():
    return {
        "percent": RE_PERCENT,
        "range":   RE_RANGE,
        "currency":RE_CURRENCY,
        "ordinal": RE_ORDINAL,
        "hhmmss":  RE_HHMMSS,
        "year":    RE_YEARN,
        "month":   RE_MONTH,
        "day":     RE_DAY,
        "longnum": RE_LONGNUM,
        "phone11": RE_PHONE11,
        "purenum": RE_PURENUM,
    }

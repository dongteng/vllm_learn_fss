"""
@Author    : zhjm
@Time      : 2026/2/7 
@File      : mock_compile.py
@Desc      : 
"""
import torch
import torch.fx as fx
import dataclasses
from typing import Any,Callable,List
from contextlib import contextmanager
import torch.nn as nn

from torch.library import Library
try:
    vllm_lib = Library("vllm", "DEF")
    vllm_lib.define("unified_attention(Tensor self) -> Tensor")
    def unified_attention_impl(self): return self
    vllm_lib.impl("unified_attention", unified_attention_impl, "CPU")
except Exception:
    # If run multiple times in the same session, this might fail. Ignore.
    pass
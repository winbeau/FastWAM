"""Sparse-acceleration case study for FastWAM (optional-idm branch: FastWAM-IDM)."""

import time

import torch

from .config import CaseStudyConfig, call_seed, parse_config
from .idm_runner import IDMCaseStudyRunner


def make_runner(model, config, prompt_store=None):
    """Case-study runner for the loaded model class."""
    name = type(model).__name__
    if name in ("FastWAMOptionalIDM", "FastWAMIDM"):
        return IDMCaseStudyRunner(model, config, prompt_store=prompt_store)
    raise TypeError(f"no case-study runner for {name} on this branch")


def timed(fn, *args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


__all__ = ["CaseStudyConfig", "IDMCaseStudyRunner", "call_seed", "make_runner", "parse_config", "timed"]

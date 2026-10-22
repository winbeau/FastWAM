"""Sparse-acceleration case study for FastWAM (CVPR'27 Sparse-rethinking)."""

from .config import CaseStudyConfig, call_seed, parse_config
from .joint_runner import JointCaseStudyRunner, timed


def make_runner(model, config, prompt_store=None):
    """Case-study runner for the loaded model class (FastWAM-IDM lives on branch optional-idm)."""
    name = type(model).__name__
    if name == "FastWAMJoint":
        return JointCaseStudyRunner(model, config, prompt_store=prompt_store)
    raise TypeError(f"no case-study runner for {name} on this branch")


__all__ = ["CaseStudyConfig", "JointCaseStudyRunner", "call_seed", "make_runner", "parse_config", "timed"]

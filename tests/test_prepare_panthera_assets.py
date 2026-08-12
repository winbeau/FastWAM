from __future__ import annotations

from pathlib import Path

import torch

from scripts.prepare_panthera_assets import validate_text_cache


def test_validate_text_cache_requires_loadable_finite_context_and_mask(tmp_path: Path) -> None:
    path = tmp_path / "cache.pt"
    path.write_bytes(b"not-a-torch-cache")
    valid, error = validate_text_cache(path, context_len=4)
    assert not valid
    assert error and "cannot load" in error

    torch.save({"context": torch.zeros(4, 8), "mask": torch.ones(4)}, path)
    assert validate_text_cache(path, context_len=4) == (True, None)

    torch.save({"context": torch.zeros(3, 8), "mask": torch.ones(4)}, path)
    valid, error = validate_text_cache(path, context_len=4)
    assert not valid
    assert error and "context must have shape" in error

    context = torch.zeros(4, 8)
    context[0, 0] = float("nan")
    torch.save({"context": context, "mask": torch.ones(4)}, path)
    valid, error = validate_text_cache(path, context_len=4)
    assert not valid
    assert error == "context contains NaN or Inf"

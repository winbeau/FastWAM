"""CPU checks of the diff selections (``prune.selection: diff | diff_global``)."""

import pytest
import torch

from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config

GRID = sel.Grid(frames=3, height=7, width=14)


def test_diff_scores_compare_each_future_token_with_its_current_frame_position():
    g = torch.Generator().manual_seed(0)
    z0 = torch.randn(GRID.per_frame, 192, generator=g)
    x0 = z0.repeat(2, 1).clone()  # a future identical to the current frame: distance 0 everywhere
    assert torch.equal(sel.diff_scores(x0, z0, GRID), torch.zeros(GRID.num_future))
    x0[98 + 3 * 14 + 5] += 2.0  # frame 2, h 3, w 5
    d = sel.diff_scores(x0, z0, GRID)
    assert int(d.argmax()) == 98 + 3 * 14 + 5 and torch.isclose(d.max(), torch.tensor(2.0 * 192 ** 0.5))


def test_per_view_topk_splits_the_budget_by_camera():
    g = torch.Generator().manual_seed(1)
    score = torch.rand(GRID.num_future, generator=g)
    third, wrist = GRID.future_view_masks("cpu")
    for k in (39, 98, 196):
        kept = sel.per_view_topk(score, k, GRID)
        assert kept.numel() == k and kept.unique().numel() == k
        assert int(third[kept].sum()) == (k + 1) // 2 and int(wrist[kept].sum()) == k // 2
        top_third = torch.topk(torch.where(third, score, torch.tensor(-1.0)), (k + 1) // 2).indices
        assert set(kept[third[kept]].tolist()) == set(top_third.tolist())
    # the third-person camera is the left half of every row (columns 0-6)
    assert bool(third[0]) and not bool(third[7]) and bool(third[98 + 6]) and not bool(third[98 + 13])


def test_config():
    cfg = parse_config({"method": "prune", "record_future_attn": True,
                        "prune": {"selection": "diff", "keep_ratio": 0.2, "warmup_steps": 1}})
    assert cfg.prune.selection == "diff" and cfg.record_future_attn
    assert parse_config({"method": "prune", "prune": {"selection": "diff_global", "diff_debug_av": True}}).prune.diff_debug_av
    for bad in ({"selection": "av", "diff_debug_av": True}, {"selection": "diff", "warmup_steps": 2},
                {"selection": "diff", "kept_index": [1, 2]}):
        with pytest.raises(ValueError):
            parse_config({"method": "prune", "prune": bad})
    with pytest.raises(ValueError):
        parse_config({"method": "prune", "record_future_attn": True,
                      "prune": {"selection": "av", "action_full_kv": True, "video_full_kv": True}})

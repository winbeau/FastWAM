"""body-select (007): fastwam.casestudy.body_fk (a verbatim copy of action-eval scripts/casestudy/body_fk.py, the online masks)."""

import inspect
import math

import numpy as np

from fastwam.casestudy import body_fk as bf


def _calib():
    # a camera looking down -z from (0, 0, 2) onto the xy plane: pixel = f * (x, y) / depth + centre, upright rows = y
    f, res = 100.0, 256
    m = np.array([[f, 0, res / 2, 0], [0, f, res / 2, 0], [0, 0, 1, 0]], dtype=float) @ np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 2], [0, 0, 0, 1]], dtype=float)
    return {"camera": {"matrix": m.tolist(), "res": res, "flip": "none"},
            "root": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]},
            "bodies": [{"name": "b0", "parent": -1, "pos": [0, 0, 0], "quat": [1, 0, 0, 0],
                        "joints": [{"type": "hinge", "axis": [0, 0, 1], "pos": [0, 0, 0], "q": 0, "q0": 0.0}]},
                       {"name": "b1", "parent": 0, "pos": [0.5, 0, 0], "quat": [1, 0, 0, 0],
                        "joints": [{"type": "slide", "axis": [1, 0, 0], "pos": [0, 0, 0], "q": 1, "q0": 0.0}]}],
            "points": [{"body": 1, "xyz": [[0, 0, 0]]}], "q_index": [0, 1]}


def test_mask_inputs_are_q_and_calib_only():
    assert list(inspect.signature(bf.body_mask).parameters) == ["q", "calib"]
    c = _calib()
    a = bf.body_mask(np.array([0.3, 0.1]), c)
    b = bf.body_mask(np.array([0.3, 0.1]), c)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_forward_hinge_and_slide():
    c = _calib()
    pos, rot = bf.forward(np.array([math.pi / 2, 0.25]), c)
    assert np.allclose(pos[1], [0, 0.75, 0], atol=1e-12)  # rotated 90 deg about z, then slid 0.25 along the rotated x
    assert np.allclose(rot[1], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    c["bodies"][0]["joints"][0]["q0"] = math.pi / 2  # MuJoCo turns by q - q0
    assert np.allclose(bf.forward(np.array([math.pi / 2, 0.0]), c)[0][1], [0.5, 0, 0], atol=1e-12)


def test_projection_tokens_and_dilation():
    c = _calib()
    xy = bf.project(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]), c["camera"])
    assert np.allclose(xy[0], [128, 127])  # the centre (upright row 128 -> raw row 255 - 128)
    assert np.all(np.isnan(xy[1]))  # behind the camera
    cov = bf.token_cover(np.array([[0.0, 0.0], [255.0, 255.0], [300.0, 10.0]]), 256)
    assert cov[0, 0] and cov[6, 6] and cov.sum() == 2
    m = np.zeros((7, 7), dtype=bool)
    m[0, 0] = m[3, 3] = True
    d = bf.dilate(m)
    assert d.sum() == 4 + 9 and d[1, 1] and not d[2, 0] and d[4, 4]
    assert bf.near_rows(m) == [0, 3 * 14 + 3]
    assert bf.iou(m, d) == 2 / 13

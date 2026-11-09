"""body-select round (2026-11-09): the agentview body mask from the robot's joint values and a fixed calibration only.

``body_mask(q, calib)`` is the whole method: forward kinematics of the robot's kinematic tree (``robot0_*`` / ``gripper0_*`` bodies,
static offsets and joint axes read once from the simulator model, like a URDF), the visible geoms' surface points carried along,
projected with the agentview camera's world -> pixel matrix onto the observation (256 x 256, flipped as the benchmark flips the raw
render), resized to the model's 224 x 224 view and binned into 32 x 32 tokens (7 x 7). A token is covered when at least one point
lands in it (no occlusion); the body set is the covered set dilated by one token (3 x 3). No image and no model output enters.

The calibration (``calibration(sim, ...)`` in ``body_masks.py``) is a JSON-able dict:
  camera     {"matrix": 3 x 4 / 4 x 4 world -> pixel on the upright render, "res": 256, "flip": "ud_lr"}
  bodies     [{"name", "parent" (index into bodies or -1 = the root pose), "pos", "quat" (wxyz), "joints": [{"type": "hinge" |
             "slide", "axis", "pos", "q": index into q, "q0"}]}] in tree order (parents first)
  root       {"pos", "quat"}: the world pose of the first body's parent
  points     [{"body": index, "xyz": [[x, y, z], ...]}] surface points in the body frame
  q_index    the state's qpos addresses of q (flattened sim state = [time, qpos, qvel])
"""

from __future__ import annotations

import numpy as np

MODEL_RES, TOKEN, GRID = 224, 32, 7


def quat_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    return np.array([[1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
                     [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
                     [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)]])


def axis_angle(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def forward(q: np.ndarray, calib: dict) -> tuple[np.ndarray, np.ndarray]:
    """World positions [n, 3] and rotations [n, 3, 3] of the calibration's bodies for joint values q (MuJoCo's kinematics: the
    body frame from its parent and static offset, then each joint about its anchor in that frame, angle q - q0)."""
    q = np.asarray(q, dtype=np.float64)
    n = len(calib["bodies"])
    pos, rot = np.zeros((n, 3)), np.zeros((n, 3, 3))
    root_p, root_r = np.asarray(calib["root"]["pos"], dtype=np.float64), quat_mat(calib["root"]["quat"])
    for i, b in enumerate(calib["bodies"]):
        pp, pr = (root_p, root_r) if b["parent"] < 0 else (pos[b["parent"]], rot[b["parent"]])
        p = pp + pr @ np.asarray(b["pos"], dtype=np.float64)
        r = pr @ quat_mat(b["quat"])
        for j in b["joints"]:
            v = float(q[j["q"]]) - float(j["q0"])
            if j["type"] == "hinge":
                anchor = p + r @ np.asarray(j["pos"], dtype=np.float64)
                rj = axis_angle(r @ np.asarray(j["axis"], dtype=np.float64), v)
                p = anchor + rj @ (p - anchor)
                r = rj @ r
            elif j["type"] == "slide":
                p = p + (r @ np.asarray(j["axis"], dtype=np.float64)) * v
            else:
                raise ValueError(f"body_fk: joint type {j['type']!r}")
        pos[i], rot[i] = p, r
    return pos, rot


def world_points(q: np.ndarray, calib: dict) -> np.ndarray:
    """Every calibration surface point in world coordinates [m, 3]."""
    pos, rot = forward(q, calib)
    out = [pos[g["body"]] + np.asarray(g["xyz"], dtype=np.float64) @ rot[g["body"]].T for g in calib["points"] if len(g["xyz"])]
    return np.concatenate(out) if out else np.zeros((0, 3))


def project(points: np.ndarray, camera: dict) -> np.ndarray:
    """World points -> observation pixels (x from the left, y from the top) [m, 2]; robosuite's matrix gives (row, col) on the
    upright image (the raw render flipped up-down), the observation is camera["flip"] of the raw render."""
    m = np.asarray(camera["matrix"], dtype=np.float64)
    res = int(camera["res"])
    ph = np.c_[points, np.ones(len(points))] @ m[:3].T
    z = ph[:, 2]
    row, col = ph[:, 1] / z, ph[:, 0] / z
    x, y = col, res - 1 - row  # upright -> raw render
    if camera["flip"] in ("ud", "ud_lr"):
        y = res - 1 - y
    if camera["flip"] in ("lr", "ud_lr"):
        x = res - 1 - x
    xy = np.c_[x, y]
    xy[z <= 0] = np.nan  # behind the camera
    return xy


def token_cover(xy: np.ndarray, res: int) -> np.ndarray:
    """Pixels on the res x res observation -> the 7 x 7 tokens of the model's 224 x 224 view that some pixel lands in (bool)."""
    out = np.zeros((GRID, GRID), dtype=bool)
    ok = np.all(np.isfinite(xy), axis=1)
    xy = xy[ok]
    cell = res / GRID  # 224 / 7 = 32 model pixels = res / 7 observation pixels
    cx, cy = np.floor((xy[:, 0] + 0.5) / cell).astype(int), np.floor((xy[:, 1] + 0.5) / cell).astype(int)
    keep = (cx >= 0) & (cx < GRID) & (cy >= 0) & (cy < GRID)
    out[cy[keep], cx[keep]] = True
    return out


def dilate(m: np.ndarray) -> np.ndarray:
    """3 x 3 (8-neighbour) dilation inside the 7 x 7 grid."""
    m = np.asarray(m, dtype=bool)
    p = np.pad(m, 1)
    out = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= p[1 + dy:1 + dy + m.shape[0], 1 + dx:1 + dx + m.shape[1]]
    return out


def body_mask(q: np.ndarray, calib: dict) -> tuple[np.ndarray, np.ndarray]:
    """(covered, body) 7 x 7 bool masks of the agentview from joint values q and the calibration only."""
    cover = token_cover(project(world_points(q, calib), calib["camera"]), int(calib["camera"]["res"]))
    return cover, dilate(cover)


def near_rows(mask: np.ndarray, width: int = 14) -> list[int]:
    """A 7 x 7 agentview mask -> the near frame's token indices r * width + c (agentview = the left half)."""
    rr, cc = np.nonzero(np.asarray(mask, dtype=bool))
    return sorted(int(r) * width + int(c) for r, c in zip(rr, cc))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else float("nan")

"""share_case_study (2026-11-25) analysis: checks, results.csv, fits and verdicts of DECISIONS.md D / E, three figures.

    python experiments/share_case_study/analyze.py --run <phase-B run dir> --out $H/analysis/share_case_study

Needs numpy and matplotlib only (action-eval venv).
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path

import numpy as np

EXEC, M_PER_UNIT = 10, 0.05
FI_DENSE_Y = {0: 0.43100846002461185, 1: 0.43529907263423573, 2: 0.40817467351466513}  # h200 future-inject, object t005-i025 call 0
T975 = {23: 2.0687}
REPS = 2000
C1 = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
CMID = (-1.0, -0.5, 0.0, 0.5, 1.0)
S_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def f32(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32)


def load(run: Path) -> tuple[dict, dict]:
    for acc in sorted(run.glob("episodes/*/accepted.json")):
        att = acc.parent / "attempts" / f"{int(json.loads(acc.read_text())['attempt']):04d}"
        for c in json.loads((att / "policy-calls.json").read_text()):
            d = (c.get("metadata") or {}).get("diagnostics") or {}
            if "share_case" in d:
                return d, d["share_case"]
    raise SystemExit(f"no share_case diagnostics under {run}")


def cos(u, v):
    n = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(u @ v / n) if n > 0 else float("nan")


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    b = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    a = float(y.mean() - b * x.mean())
    res = y - a - b * x
    s2 = float(res @ res) / (n - 2)
    sxx = float(((x - x.mean()) ** 2).sum())
    se_b = (s2 / sxx) ** 0.5
    se_a = (s2 * (1 / n + x.mean() ** 2 / sxx)) ** 0.5
    t = T975.get(n - 2, 1.96)
    r2 = 1 - float(res @ res) / float(((y - y.mean()) ** 2).sum())
    return {"a": a, "b": b, "a_ci": [a - t * se_a, a + t * se_a], "b_ci": [b - t * se_b, b + t * se_b], "r2": r2, "n": n}


def rows_of(sc: dict) -> list[dict]:
    ent = sc["entries"]
    L, H = int(sc["layers"]), int(sc["heads"])
    eef = np.asarray(sc["eef"])
    xs = {s: float(np.linalg.norm(eef[int(round(10 * s))] - eef[0])) for s in S_GRID}
    rows = []
    for key, e in ent.items():
        name, j = key.split("@")
        j = int(j)
        dense = ent[f"dense@{j}"]
        cont, dcont = f32(e["cont_actions_f32"]).reshape(-1, 7), f32(dense["cont_actions_f32"]).reshape(-1, 7)
        D, Dd = cont[:EXEC, :3].astype(float).sum(0), dcont[:EXEC, :3].astype(float).sum(0)
        D6, Dd6 = cont[:EXEC, :6].astype(float).sum(0), dcont[:EXEC, :6].astype(float).sum(0)
        ml = f32(e["m_last_f32"]).reshape(L, H)
        ms = np.asarray(e["m_steps"])
        s = e.get("s")
        row = {"cond": name, "exp": e["exp"], "path": e["path"], "c": float(e.get("c") or 0.0), "c_eff": e.get("c_eff"),
               "scope": e.get("scope", "all"), "s": s, "sparse": bool(e.get("sparse", False)), "ht": bool(e.get("ht", False)),
               "seed": j, "m": float(ml.mean()), "m_steps1_9": float(ms[1:].mean()), "r": float(np.linalg.norm(D) / np.linalg.norm(Dd)),
               "dq_norm_m": M_PER_UNIT * float(np.linalg.norm(D)), "cos": cos(D, Dd), "cos6": cos(D6, Dd6),
               "x_m": xs[s] if s is not None else None, "x_tilde": xs[s] / xs[1.0] if s is not None else None,
               "masked_calls": e["masked_calls"], "raw_maxabs_vs_dense": float(np.abs(f32(e["raw_actions_f32"]) - f32(dense["raw_actions_f32"])).max())}
        for li in range(L):
            row[f"m_L{li:02d}"] = float(ml[li].mean())
        rows.append(row)
    return rows


def pick(rows, **kw):
    return [r for r in rows if all((r[k] == v) if not isinstance(v, float) else (r[k] is not None and abs(r[k] - v) < 1e-9)
                                   for k, v in kw.items())]


def mean_sd(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def groups2(rows):
    g = {}
    for c in (-1.0, 0.0, 1.0):
        g[f"c={c:+g}"] = [r for r in rows if r["exp"] == 2 and not r["sparse"] and abs(r["c"] - c) < 1e-9]
    g["sparse"] = [r for r in rows if r["exp"] == 2 and r["sparse"] and not r["ht"]]
    g["sparse+HT"] = [r for r in rows if r["exp"] == 2 and r["sparse"] and r["ht"]]
    return g


def boot_fit(gr: dict, seeds: list[int], rng) -> dict:
    out = {k: {"a": [], "b": []} for k in gr}
    for _ in range(REPS):
        pick_s = rng.choice(seeds, size=len(seeds), replace=True)
        for k, rs in gr.items():
            x = np.concatenate([[r["x_tilde"] for r in rs if r["seed"] == s] for s in pick_s])
            y = np.concatenate([[r["r"] for r in rs if r["seed"] == s] for s in pick_s])
            if np.var(x) == 0:
                continue
            f = ols(x, y) if len(x) > 2 else None
            if f:
                out[k]["a"].append(f["a"])
                out[k]["b"].append(f["b"])
    return out


def analyse(diag: dict, sc: dict, rows: list[dict]) -> dict:
    seeds = sorted({r["seed"] for r in rows})
    res: dict = {"seeds": seeds, "select_layer": sc["select_layer"], "c_ht": sc["c_ht"], "c_ht_bf16": sc["c_ht_bf16"],
                 "s_valid": sc["s_valid"], "x_m": {f"{s:g}": float(np.linalg.norm(np.asarray(sc['eef'])[int(round(10 * s))] - np.asarray(sc['eef'])[0])) for s in S_GRID}}
    # ---- checks
    zero = pick(rows, cond="e1_c0_all")
    k1 = {"executed_equal_dense0": sc["executed_equal_dense0"], "frame_equal": sc["frame_equal"],
          "zero_mask_r": [r["r"] for r in zero], "zero_mask_cos": [r["cos"] for r in zero],
          "zero_mask_raw_maxabs": max(r["raw_maxabs_vs_dense"] for r in zero)}
    k1["pass"] = bool(k1["executed_equal_dense0"] and k1["frame_equal"] and all(0.99 <= x <= 1.01 for x in k1["zero_mask_r"])
                      and all(x >= 0.999 for x in k1["zero_mask_cos"]))
    dense = pick(rows, cond="dense")
    k1["dense_y_vs_h200"] = {j: {"h100": r["dq_norm_m"], "h200": FI_DENSE_Y[j], "diff": r["dq_norm_m"] - FI_DENSE_Y[j]}
                             for r in dense for j in [r["seed"]] if j in FI_DENSE_Y}
    s0, s0ht = pick(rows, cond="s0"), pick(rows, cond="s0ht")
    k2 = {"m_dense_step9": mean_sd([r["m"] for r in dense]), "m_s0_step9": mean_sd([r["m"] for r in s0]),
          "m_dense_steps1_9": mean_sd([r["m_steps1_9"] for r in dense]), "m_s0_steps1_9": mean_sd([r["m_steps1_9"] for r in s0]),
          "r_s0": mean_sd([r["r"] for r in s0])}
    k3 = {"m_s0ht_step9": mean_sd([r["m"] for r in s0ht]), "m_s0ht_steps1_9": mean_sd([r["m_steps1_9"] for r in s0ht]),
          "r_s0ht": mean_sd([r["r"] for r in s0ht]), "cos_s0ht": mean_sd([r["cos"] for r in s0ht])}
    k3["close_to_dense"] = abs(k3["m_s0ht_step9"][0] - k2["m_dense_step9"][0]) <= 0.05
    res["checks"] = {"K1": k1, "K2": k2, "K3": k3}
    # ---- prediction 1
    tab = {}
    for scope in ("all", "last"):
        tab[scope] = {}
        for c in C1 + (sc["c_ht"],):
            rs = dense if c == 0.0 and scope == "last" else [r for r in rows if r["exp"] == 1 and not r["sparse"] and r["scope"] == scope and abs(r["c"] - c) < 1e-9]
            if c == 0.0 and scope == "all":
                rs = zero
            if not rs:
                continue
            tab[scope][f"{c:g}"] = {"m": mean_sd([r["m"] for r in rs]), "m_steps1_9": mean_sd([r["m_steps1_9"] for r in rs]),
                                    "r": mean_sd([r["r"] for r in rs]), "cos": mean_sd([r["cos"] for r in rs]),
                                    "cos6": mean_sd([r["cos6"] for r in rs]), "dq": mean_sd([r["dq_norm_m"] for r in rs]),
                                    "r_seeds": [r["r"] for r in sorted(rs, key=lambda r: r["seed"])]}
    a = tab["all"]
    m_seq = [a[f"{c:g}"]["m"][0] for c in C1]
    r_mid = [a[f"{c:g}"]["r"][0] for c in CMID]
    p1 = {"m_monotone": bool(all(np.diff(m_seq) > 0)), "r_monotone_mid": bool(all(np.diff(r_mid) > 0)),
          "r_below_neg": {f"{c:g}": sum(x < 1 for x in a[f"{c:g}"]["r_seeds"]) for c in (-1.0, -0.5)},
          "r_above_pos": {f"{c:g}": sum(x > 1 for x in a[f"{c:g}"]["r_seeds"]) for c in (0.5, 1.0)},
          "cos_min_mid": min(a[f"{c:g}"]["cos"][0] for c in CMID)}
    n = len(seeds)
    p1["side_ok"] = all(v == n for v in p1["r_below_neg"].values()) and all(v == n for v in p1["r_above_pos"].values())
    p1["holds"] = bool(p1["m_monotone"] and p1["r_monotone_mid"] and p1["side_ok"] and p1["cos_min_mid"] >= 0.98)
    rf = {c: a[f"{c:g}"]["r"][0] for c in C1}
    p1["r_monotone_full"] = bool(all(np.diff([rf[c] for c in C1]) > 0))
    p1["saturation_pos"] = bool((rf[3.0] - rf[2.0]) < (rf[1.0] - rf[0.0]) / 3)
    p1["saturation_neg"] = bool((rf[-2.0] - rf[-3.0]) < (rf[0.0] - rf[-1.0]) / 3)
    p1["unstable"] = [f"{c:g}" for c in C1 if a[f"{c:g}"]["cos"][0] < 0.95]
    # per-layer monotonicity of m in c
    Ls = int(sc["layers"])
    lay = []
    for li in range(Ls):
        seq = [np.mean([r[f"m_L{li:02d}"] for r in (zero if c == 0.0 else pick(rows, exp=1, scope="all", c=c, sparse=False))]) for c in C1]
        lay.append({"layer": li, "monotone": bool(all(np.diff(seq) > 0)), "m_c0": float(seq[4]), "dm_pm1": float(seq[6] - seq[2])})
    p1["layers_monotone"] = sum(x["monotone"] for x in lay)
    res["p1"] = {"table": tab, "verdict": p1, "layers": lay}
    # ---- prediction 2
    gr = groups2(rows)
    fits = {k: ols([r["x_tilde"] for r in rs], [r["r"] for r in rs]) for k, rs in gr.items()}
    rng = np.random.default_rng(0)
    bt = boot_fit(gr, seeds, rng)
    for k in fits:
        fits[k]["a_boot"] = list(np.percentile(bt[k]["a"], [2.5, 97.5]))
        fits[k]["b_boot"] = list(np.percentile(bt[k]["b"], [2.5, 97.5]))
    db = np.asarray(bt["c=+1"]["b"]) - np.asarray(bt["c=-1"]["b"])
    da = np.asarray(bt["c=+1"]["a"]) - np.asarray(bt["c=-1"]["a"])
    xg = sorted({r["x_tilde"] for r in gr["c=+0"]})

    def dist(g1, g2):
        return float(np.mean([abs((fits[g1]["a"] + fits[g1]["b"] * x) - (fits[g2]["a"] + fits[g2]["b"] * x)) for x in xg]))

    p2 = {"b_order": bool(fits["c=-1"]["b"] < fits["c=+0"]["b"] < fits["c=+1"]["b"]),
          "db": fits["c=+1"]["b"] - fits["c=-1"]["b"], "db_ci": list(np.percentile(db, [2.5, 97.5])),
          "da": fits["c=+1"]["a"] - fits["c=-1"]["a"], "da_ci": list(np.percentile(da, [2.5, 97.5])),
          "dist": {f"{g}|{h}": dist(g, h) for g in ("sparse", "sparse+HT") for h in ("c=-1", "c=+0", "c=+1")}}
    p2["slope_part"] = bool(p2["b_order"] and p2["db_ci"][0] > 0)
    p2["intercept_same"] = bool((p2["da_ci"][0] <= 0 <= p2["da_ci"][1]) or abs(p2["da"]) <= 0.05)
    p2["sparse_near_down"] = p2["dist"]["sparse|c=-1"] < p2["dist"]["sparse|c=+0"]
    p2["sparseht_near_dense"] = p2["dist"]["sparse+HT|c=+0"] < min(p2["dist"]["sparse+HT|c=-1"], p2["dist"]["sparse+HT|c=+1"])
    p2["m"] = {k: mean_sd([r["m"] for r in rs if r["s"] == 1.0]) for k, rs in gr.items()}
    res["p2"] = {"fits": fits, "verdict": p2}
    # ---- prediction 3
    sens = {}
    for s in (0.0, 1.0, 2.0):
        rs = [r for r in rows if r["exp"] == 2 and not r["sparse"] and r["s"] == s and any(abs(r["c"] - c) < 1e-9 for c in CMID)]
        f = ols([r["c"] for r in rs], [r["r"] for r in rs])
        r0 = np.mean([r["r"] for r in rs if r["c"] == 0.0])
        rng_ = [next(r["r"] for r in rs if r["seed"] == j and r["c"] == 1.0) - next(r["r"] for r in rs if r["seed"] == j and r["c"] == -1.0) for j in seeds]
        full = {f"{c:g}": mean_sd([r["r"] for r in rows if r["exp"] == 2 and not r["sparse"] and r["s"] == s and abs(r["c"] - c) < 1e-9]) for c in C1}
        sens[f"{s:g}"] = {"S": f["b"], "S_ci": f["b_ci"], "range_pm1": mean_sd(rng_), "r_c0": float(r0), "S_rel": f["b"] / float(r0),
                          "r_by_c": full, "m_by_c": {f"{c:g}": mean_sd([r["m"] for r in rows if r["exp"] == 2 and not r["sparse"] and r["s"] == s and abs(r["c"] - c) < 1e-9]) for c in C1}}
    ratio = abs(sens["0"]["S"]) / sens["1"]["S"] if sens["1"]["S"] > 0 else float("nan")
    p3 = {"ratio_S0_S1": ratio, "ratio_S0_S2": abs(sens["0"]["S"]) / sens["2"]["S"] if sens["2"]["S"] > 0 else float("nan")}
    p3["verdict"] = "holds" if ratio <= 1 / 3 else ("fails: share-only fits better" if ratio >= 2 / 3 else "partial")
    lines = {k: (fits[k]["a"], fits[k]["b"]) for k in ("c=-1", "c=+0", "c=+1")}
    p3["intercept_spread"] = max(v[0] for v in lines.values()) - min(v[0] for v in lines.values())
    res["p3"] = {"sens": sens, "verdict": p3}
    return res


def figures(rows, res, out: Path, sc: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tab = res["p1"]["table"]
    cht = sc["c_ht"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for scope, mk, col in (("all", "o-", "tab:blue"), ("last", "s--", "tab:orange")):
        cs = sorted(float(c) for c in tab[scope])
        for i, key in enumerate(("m", "r", "cos")):
            mu = [tab[scope][f"{c:g}"][key][0] for c in cs]
            sd = [tab[scope][f"{c:g}"][key][1] for c in cs]
            ax[i].errorbar(cs, mu, yerr=sd, fmt=mk, color=col, ms=4, capsize=2, label=f"bias on {'every step' if scope == 'all' else 'step 9 only'}")
    k = res["checks"]
    ax[0].axhline(k["K2"]["m_s0_step9"][0], color="gray", ls=":", label="S0 (keep 20%)")
    ax[0].axhline(k["K3"]["m_s0ht_step9"][0], color="purple", ls=":", label="S0 + HT")
    ax[1].axhline(1.0, color="k", lw=0.5)
    ax[1].axhline(k["K2"]["r_s0"][0], color="gray", ls=":", label="S0")
    ax[1].axhline(k["K3"]["r_s0ht"][0], color="purple", ls=":", label="S0 + HT")
    for i, lab in enumerate(("m (step 9, action -> future share)", "r = |D| / |D_dense|", "cos(D, D_dense)")):
        ax[i].axvline(cht, color="red", lw=0.8, ls="--")
        ax[i].set_xlabel("logit bias c on action -> future keys")
        ax[i].set_ylabel(lab)
    ax[0].text(cht, ax[0].get_ylim()[0], " c = log(N/n)", color="red", fontsize=8, va="bottom")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    fig.suptitle("Prediction 1: libero_object t005-i025 call 0, dense, 5 seeds (mean ± sd)", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig1_c_sweep.{ext}", dpi=160)
    plt.close(fig)

    gr = groups2(rows)
    fits = res["p2"]["fits"]
    colors = {"c=-1": "tab:blue", "c=+0": "k", "c=+1": "tab:red", "sparse": "tab:green", "sparse+HT": "tab:purple"}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    xs = np.linspace(0, 2.4, 20)
    for kname, rs in gr.items():
        jit = {"c=-1": -0.03, "c=+0": -0.015, "c=+1": 0.0, "sparse": 0.015, "sparse+HT": 0.03}[kname]
        ax.scatter([r["x_tilde"] + jit for r in rs], [r["r"] for r in rs], s=10, alpha=0.6, color=colors[kname])
        f = fits[kname]
        ax.plot(xs, f["a"] + f["b"] * xs, color=colors[kname], label=f"{kname}: r = {f['a']:.2f} + {f['b']:.2f} x~")
    ax.plot(xs, xs, "k:", lw=0.6)
    ax.set_xlabel("x~ : EE displacement in the injected future / at s = 1")
    ax.set_ylabel("r = action displacement / dense")
    ax.legend(fontsize=8)
    ax.set_title("Prediction 2: r vs injected future (5 s x 5 seeds per line)", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig2_r_vs_x.{ext}", dpi=160)
    plt.close(fig)

    sens = res["p3"]["sens"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for s, col in (("0", "tab:gray"), ("1", "tab:blue"), ("2", "tab:red")):
        cs = sorted(float(c) for c in sens[s]["r_by_c"])
        ax[0].errorbar(cs, [sens[s]["r_by_c"][f"{c:g}"][0] for c in cs], yerr=[sens[s]["r_by_c"][f"{c:g}"][1] for c in cs], fmt="o-",
                       color=col, ms=4, capsize=2, label=f"s = {s}: slope on [-1, 1] = {sens[s]['S']:.3f}")
        ax[1].errorbar(cs, [sens[s]["r_by_c"][f"{c:g}"][0] / sens[s]["r_c0"] for c in cs], fmt="o-", color=col, ms=4, label=f"s = {s}")
    ax[0].set_xlabel("c")
    ax[0].set_ylabel("r")
    ax[0].legend(fontsize=8)
    ax[0].set_title("absolute")
    ax[1].set_xlabel("c")
    ax[1].set_ylabel("r / r(c = 0)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("relative to c = 0")
    fig.suptitle("Prediction 3: sensitivity of r to c when the future equals the present (s = 0) vs s = 1, 2", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig3_sensitivity.{ext}", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    diag, sc = load(Path(args.run))
    rows = rows_of(sc)
    res = analyse(diag, sc, rows)
    (out / "checks.json").write_text(json.dumps(res["checks"], indent=1) + "\n")
    (out / "analysis.json").write_text(json.dumps(res, indent=1) + "\n")
    keys = list(rows[0].keys())
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["exp"], r["cond"], r["seed"])):
            w.writerow(r)
    L, H = int(sc["layers"]), int(sc["heads"])
    np.savez_compressed(out / "m_heads.npz", names=np.asarray([k for k in sc["entries"]]),
                        m_last=np.stack([f32(e["m_last_f32"]).reshape(L, H) for e in sc["entries"].values()]))
    figures(rows, res, out, sc)
    print(json.dumps({"K1_pass": res["checks"]["K1"]["pass"], "p1": res["p1"]["verdict"], "p2": res["p2"]["verdict"],
                      "p3": res["p3"]["verdict"]}, indent=1))


if __name__ == "__main__":
    main()

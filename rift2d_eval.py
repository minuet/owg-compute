"""Verdicts of the 2D rift runs of tools/aspect_rift2d.py (PARAM_AUDIT 11.17, criteria fixed before running): a run
rifts at the first time its opening velocity (maximum velocity on the right boundary) exceeds V_FAST, and has failed if
it reaches T_END without; otherwise it is undecided and chain.py continues it. Per structure, the 2D force F20 that
rifts in T_TARGET is interpolated in log time between the two runs around T_TARGET, and compared with the 1D critical
force of the rift criterion (shells_rheology at the critical strain rate, PARAM_AUDIT 11.13).
Also prints the measured force range (as brune_m1_eval.py) to check the force boundary.
Usage: python -I rift2d_eval.py <results dir> [<results dir> ...]"""
import glob, json, math, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]       # brune_m1_eval here, shells_rheology in tools/
from brune_m1_eval import V_FAST, T_FAIL as T_END      # 10 mm/yr; 30 Myr, as in the M1 reproduction (PARAM_AUDIT 11.15)

T_TARGET = 20.0                    # Myr, force that rifts in 20 Myr (E-1121-18; Brune et al. 2016 Ext. Data Fig. 7)
EDOT_CRIT = 1e-17                  # 1/s, critical strain rate of the rift criterion (PARAM_AUDIT 11.13)


def load(d):
    run = json.load(open(glob.glob(f"{d}/*_run.json")[0]))
    st = glob.glob(f"{d}/*_statistics")
    if not st:
        return run, None, None
    head = [l for l in open(st[0]) if l.startswith("#")]
    col = lambda name: next(int(l[1:].split(":")[0]) - 1 for l in head if name in l)
    s = np.loadtxt(st[0], usecols=(col("Time (years)"), col('Maximum velocity magnitude on boundary with indicator 1')))
    s = np.atleast_2d(s)
    f = [float(m) for m in re.findall(r"F_dev_TN_m (\S+)", open(glob.glob(f"{d}/*_log.txt")[0], errors="replace").read())]
    return run, s, np.array(f)


def verdict(d):
    run, s, f = load(d)
    out = {"q": run["q_mW"], "zc": run["zc_km"], "zm": run["zm_km"], "F": run["F_TN_m"], "dir": d}
    if s is None:
        return {**out, "verdict": "NO DATA", "t_end": 0.0, "t_fast": None}
    t, v = s[:, 0] / 1e6, s[:, 1]
    fast = np.nonzero(v > V_FAST)[0]
    out.update(t_end=float(t[-1]), t_fast=float(t[fast[0]]) if len(fast) else None,
               F_meas=(float(np.percentile(f, 5)), float(np.percentile(f, 95))) if len(f) else None)
    out["verdict"] = "RIFT" if len(fast) else "NO RIFT" if t[-1] >= T_END else "UNDECIDED"
    return out


def f20(runs):
    """F at T_TARGET from the runs of one structure: log-linear between the nearest runs on either side, or a bracket"""
    r = sorted(runs, key=lambda x: x["F"])
    for a, b in zip(r, r[1:]):
        ta = a["t_fast"] if a["verdict"] == "RIFT" else math.inf
        tb = b["t_fast"] if b["verdict"] == "RIFT" else math.inf
        if ta >= T_TARGET > tb:
            if math.isinf(ta):
                return None, (a["F"], b["F"])
            w = (math.log(T_TARGET) - math.log(ta)) / (math.log(tb) - math.log(ta))
            return a["F"] + w * (b["F"] - a["F"]), (a["F"], b["F"])
    return None, None


def main(dirs):
    import shells_rheology as sr
    res = [verdict(d) for d in dirs]
    for r in sorted(res, key=lambda x: (x["q"], x["zc"], x["F"])):
        fm = f"{r['F_meas'][0]:.2f}..{r['F_meas'][1]:.2f}" if r.get("F_meas") else "-"
        print(f"q {r['q']:g} zc {r['zc']:g} zm {r['zm']:g}  F {r['F']:6.3f}  reached {r['t_end']:5.2f} Myr  "
              f"fast at {r['t_fast']}  F measured p5..p95 {fm}  {r['verdict']}")
    groups = {}
    for r in res:
        groups.setdefault((r["q"], r["zc"], r["zm"]), []).append(r)
    for (q, zc, zm), g in sorted(groups.items()):
        fc = sr.shells_strength(q * 1e-3, zc * 1e3, zm * 1e3, EDOT_CRIT) / 1e12
        f, br = f20(g)
        print(f"structure q {q:g} zc {zc:g} zm {zm:g}: 1D Fc {fc:.3f} TN/m; 2D F20 " +
              (f"{f:.3f} (between {br[0]:g} and {br[1]:g}), F20 / Fc = {f / fc:.3f}" if f else
               f"between {br[0]:g} and {br[1]:g} (no rifted run above {T_TARGET:g} Myr yet)" if br else "not bracketed yet"))
    return res


if __name__ == "__main__":
    main(sys.argv[1:])

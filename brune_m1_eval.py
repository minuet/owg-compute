"""Judge the M1 reproduction runs of tools/aspect_brune.py against the criteria fixed before running (PARAM_AUDIT 11.15,
E-1121-26): opening velocity (maximum velocity on the right boundary, ASPECT statistics) above V_FAST within
[T_PAPER / TOL, T_PAPER * TOL] for 9.0 and 8.6 TN/m, never above V_FAST by T_FAIL for 8.0, and the order 9.0 < 8.6 < 8.0.
Also prints, every Myr, the opening velocity, the largest surface velocity and the force the plugin measured, to check
the boundary (E-1122-01: no rocking, force near F).
Usage: python -I brune_m1_eval.py <results dir of F=9.0> [<dir of 8.6> [<dir of 8.0>]]"""
import glob, json, re, sys
import numpy as np

V_FAST = 10e-3                     # m/yr, fast phase reached (PARAM_AUDIT 11.15)
TOL = 1.5                          # time within a factor 1.5 of theirs (PARAM_AUDIT 11.15, the shape-test tolerance)
T_PAPER = {9.0: 10.8, 8.6: 20.0}   # Myr, Brune et al. 2016 Supplementary Fig. 7 (8.0: failed rift)
T_FAIL = 30.0                      # Myr, length of their velocity plots


def load(d):
    run = json.load(open(glob.glob(f"{d}/*_run.json")[0]))
    s = np.loadtxt(glob.glob(f"{d}/*_statistics")[0], usecols=(1, 19, 25))   # time, right max |v|, top max |v|
    f = [tuple(map(float, m)) for m in re.findall(r"t_yr (\S+) v_old_mm_yr \S+ F_dev_TN_m (\S+)",
                                                 open(glob.glob(f"{d}/*_log.txt")[0], errors="replace").read())]
    return run, s, np.array(f)


def main(dirs):
    rift = {}
    for d in dirs:
        run, s, f = load(d)
        F = run["F_TN_m"]
        t, vr, vt = s[:, 0] / 1e6, s[:, 1], s[:, 2]
        fast = np.nonzero(vr > V_FAST)[0]
        rift[F] = t[fast[0]] if len(fast) else None
        print(f"F = {F} TN/m: reached {t[-1]:.2f} Myr, pieces {[p['exit'] for p in run['pieces']]}")
        print("   Myr  v_right mm/yr  v_top_max mm/yr  F_measured TN/m (min..max in that Myr)")
        for m in range(int(t[-1]) + 1):
            k = np.nonzero(t >= m)[0][0]
            ff = f[(f[:, 0] >= m * 1e6) & (f[:, 0] < (m + 1) * 1e6), 1] if len(f) else []
            print(f"  {t[k]:5.2f}  {vr[k] * 1e3:12.2f}  {vt[k] * 1e3:15.2f}  " +
                  (f"{ff.min():.2f}..{ff.max():.2f}" if len(ff) else "-"))
        if F in T_PAPER:
            lo, hi = T_PAPER[F] / TOL, T_PAPER[F] * TOL
            if rift[F] is None:
                verdict = "UNDECIDED (not fast yet, run on)" if t[-1] < hi else "FAIL (no fast phase)"
            else:
                verdict = "PASS" if lo <= rift[F] <= hi else "FAIL"
            print(f"  fast phase at {rift[F]} Myr; window {lo:.1f}-{hi:.1f} Myr (paper {T_PAPER[F]}): {verdict}")
        else:
            verdict = ("FAIL (fast phase)" if rift[F] is not None else
                       "PASS" if t[-1] >= T_FAIL else "UNDECIDED (run on to 30 Myr)")
            print(f"  fast phase at {rift[F]} Myr; paper: failed rift by {T_FAIL} Myr: {verdict}")
    known = [rift[F] if rift[F] is not None else np.inf for F in sorted(rift, reverse=True)]
    if len(known) > 1:
        print("order (stronger force rifts first):", "PASS" if all(a <= b for a, b in zip(known, known[1:])) else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1:])

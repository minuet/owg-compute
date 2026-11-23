"""Polarity of a tools/aspect_si2d.py run from its point values: which plate's crust is carried down.
A crust is "subducted" once some of it (field > 0.5) sits at least D_SUB below the initial surface on the point grid;
the verdict is the first output time at which exactly one of the two crusts is subducted (YOUNG or OLD), BOTH if the
two reach it at the same output, UNDECIDED if neither does by the last output (PARAM_AUDIT 11.19).
Usage: python si2d_eval.py <dir or point_values file> [...]"""
import glob, json, os, sys
import numpy as np

D_SUB_KM = 25.0      # technical: 3.5 crustal thicknesses, beyond flexure and compressional thickening (PARAM_AUDIT 11.19)
Y_KM = 110.0         # box height of aspect_si2d.py


def load(path):
    head = [l for l in open(path) if l.startswith("#")][0]
    names = head[1:].replace("<", " ").replace(">", " ").split()
    a = np.loadtxt(path, comments="#")
    return {n: a[:, i] for i, n in enumerate(names)}


def verdict(path):
    if os.path.isdir(path):
        path = glob.glob(f"{path}/*point_values.txt")[0]
    v = load(path)
    depth = Y_KM - v["evaluation_point_y"] / 1e3
    out = {"file": os.path.basename(path), "times_myr": [], "max_depth_young_km": [], "max_depth_old_km": []}
    for t in np.unique(v["time"]):
        k = v["time"] == t
        dy = depth[k][v["crust_y"][k] > 0.5]; do = depth[k][v["crust_o"][k] > 0.5]
        out["times_myr"].append(round(t / 1e6, 3))
        out["max_depth_young_km"].append(round(float(dy.max()), 1) if dy.size else None)
        out["max_depth_old_km"].append(round(float(do.max()), 1) if do.size else None)
    out["verdict"], out["at_myr"] = "UNDECIDED", None
    for t, y, o in zip(out["times_myr"], out["max_depth_young_km"], out["max_depth_old_km"]):
        sy, so = (y or 0) >= D_SUB_KM, (o or 0) >= D_SUB_KM
        if sy or so:
            out["verdict"], out["at_myr"] = ("BOTH" if sy and so else "YOUNG" if sy else "OLD"), t
            break
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = verdict(p)
        print(json.dumps({k: r[k] for k in ("file", "verdict", "at_myr")}), "last depths young/old km:",
              r["max_depth_young_km"][-1], r["max_depth_old_km"][-1])

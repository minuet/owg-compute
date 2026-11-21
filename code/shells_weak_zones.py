"""Weak zones of a SHELLS grid from its own deformation (E-1121-05): the continuum triangles that deform fastest in a
run with plate-boundary faults only get a weaker rheology in the next run (rheology 1, continuum friction lowered), the
way the authors' 2026 Earth model weakens its orogens, but placed by the computed strain rate instead of drawn outlines,
so that a planet's grid gets them from its own plate history and mechanics.
Strain rate of a triangle: the horizontal strain-rate tensor of the linear velocity field of its 3 nodes, in the
triangle's plane; its size is the largest principal rate by absolute value. The weak set is the fastest triangles that
together cover the area of the reference zones: on Earth the 13 PB2002 orogens (Bird 2003), the observed extent of
diffuse plate-boundary deformation; the strain-rate level reached there is the threshold to carry to a planet.
Usage: .venv/bin/python -I tools/shells_weak_zones.py <vEarth_*.22> <run Earth5R.feg> <PB2002_orogens.dig> <out.npz>"""
import sys, json
import numpy as np
import shells_grid as sg

SEC_PER_YR = 3.15576e7           # Julian year


def read_feg(fn):
    L = open(fn).read().splitlines()
    n = int(L[1].split()[0]); ne = int(L[2 + n].split()[0])
    P = sg.xyz(*np.array([[float(x) for x in l.split()[1:3]] for l in L[2:2 + n]]).T)
    tri = np.array([[int(x) - 1 for x in l.split()[1:4]] for l in L[3 + n:3 + n + ne]])
    return P, tri


def strain_rate(P, tri, vfile):
    """per triangle: largest principal horizontal strain rate by absolute value (1/yr) and the area (m^2); vfile: SHELLS
    node velocities (theta, phi components in m/s, node order of the grid)"""
    n = len(P)
    v = np.array([float(x.replace("D", "E")) for l in open(vfile).read().splitlines()[3:] for x in l.split()])[:2 * n]
    lon, lat = sg.lonlat(P)
    th, ph = np.radians(90.0 - lat), np.radians(lon)
    eth = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], 1)
    eph = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], 1)
    V = v[0::2][:, None] * eth + v[1::2][:, None] * eph
    X = P[tri] * sg.R_KM * 1e3
    e1, e2 = X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]
    nrm = np.cross(e1, e2); area = np.linalg.norm(nrm, axis=1) / 2; nrm /= (2 * area)[:, None]
    a1 = e1 / np.linalg.norm(e1, axis=1)[:, None]; a2 = np.cross(nrm, a1)
    dv1, dv2 = V[tri[:, 1]] - V[tri[:, 0]], V[tri[:, 2]] - V[tri[:, 0]]
    B = np.stack([np.stack([(e1 * a1).sum(1), (e1 * a2).sum(1)], 1), np.stack([(e2 * a1).sum(1), (e2 * a2).sum(1)], 1)], 1)
    dV = np.stack([np.stack([(dv1 * a1).sum(1), (dv1 * a2).sum(1)], 1), np.stack([(dv2 * a1).sum(1), (dv2 * a2).sum(1)], 1)], 1)
    G = np.linalg.solve(B, dV)                    # G[k, i, j] = d v_j / d x_i
    E = (G + G.transpose(0, 2, 1)) / 2
    return np.abs(np.linalg.eigvalsh(E)).max(1) * SEC_PER_YR, area


def weak_by_rate(rate, area, ref_area):
    """the fastest triangles covering ref_area; returns the mask and the strain-rate threshold"""
    order = np.argsort(-rate)
    k = np.searchsorted(np.cumsum(area[order]), ref_area)
    mask = np.zeros(len(rate), bool); mask[order[:k]] = True
    return mask, float(rate[order[k]])


def main(vfile, feg, dig, out):
    P, tri = read_feg(feg)
    rate, area = strain_rate(P, tri, vfile)
    orog = sg.weak_orogens(P, tri, dig, "all")
    mask, thr = weak_by_rate(rate, area, area[orog].sum())
    cen = P[tri].sum(1); cen /= np.linalg.norm(cen, axis=1)[:, None]
    np.savez_compressed(out, weak=mask, centroid=cen.astype(np.float32), rate=rate.astype(np.float32))
    print(json.dumps({"triangles": len(tri), "weak": int(mask.sum()), "threshold_per_yr": thr,
                      "area_share_of_sphere": round(float(area[mask].sum() / area.sum()), 4),
                      "weak_area_inside_orogens": round(float(area[mask & orog].sum() / area[mask].sum()), 3)}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])

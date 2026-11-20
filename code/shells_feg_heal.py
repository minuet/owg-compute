"""E-1119-16 test: does the authors' Earth model need the faults inside plates?
Earth5R.feg (Bird et al. 2008) has 2846 fault elements; about half lie inside plates (orogen faults taken from Earth's
fault databases), which a generated planet will not have: its grid can only carry the plate boundaries of its plate
history. This writes a copy of the grid with every fault element "healed" whose two sides belong to the same PB2002
plate, the way OrbWin's Heal command does it (the node pairs facing each other across the fault become one node), and
renumbers the nodes in the velocity boundary-condition files to match.
  - side of a fault element: SHELLS' order (MOD_Data.f90, GetNet) is n1, n2 on the near side, n3 opposite n2, n4
    opposite n1; the triangle having n1 and n2 as corners is the near side, the one having n3 and n4 the far side
  - plate of a side: the PB2002 outline (PB2002_plates.dig, the one SHELLS reads) containing the centroid of that
    triangle. Regions fenced by faults cannot be used instead: Earth5R leaves diffuse boundaries (Asian orogens, East
    Africa, India-Australia) as continuum, so one fault-free region spans up to 20 PB2002 plates
  - healed: n1 merged with n4 and n2 with n3; a merged node keeps the data of the lower-numbered copy (the copies are
    at one place, OrbData gives them the same data)
  - Earth5R's fault lines and PB2002's outlines are digitized separately; where they are a triangle apart, both
    triangles of a boundary fault can fall in one outline. Healing such a fault would join the two plates through its
    nodes and collapse the boundary faults next to it, so a fault is healed only if, after the merge, no kept fault has
    its two sides joined at both ends (checked fault by fault; the refused ones stay as faults and are counted). One
    end joined is allowed: it is a fault tip, which Earth5R already has 377 + 385 times (n1 = n4 or n2 = n3)
  - boundary conditions: SHELLS (MOD_Shells.f90, Downer) expects one condition for each footwall node of a fault
    dipping at most subDip = 19 degrees (MOD_SharedVars.f90), in increasing node order, and stops otherwise. The
    list is made again for the healed grid the same way; each listed node keeps the condition its original copy had,
    and the plate-interior conditions (type 5, Earth5R-type4AplusA.bcs) follow, renumbered
Usage: .venv/bin/python -I tools/shells_feg_heal.py <ShellSet INPUT dir> <out dir>   (writes Earth5R.feg and both .bcs)"""
import os, sys
import numpy as np
import pygplates


def outlines(fn):
    polys, name, pts = [], None, []
    for l in open(fn):
        s = l.strip()
        if not s:
            continue
        if s.startswith("***"):
            polys.append((name, pygplates.PolygonOnSphere([(la, lo) for lo, la in pts]))); pts = []
        elif s[0] in "+-0123456789":
            lo, la = (float(x) for x in s.split(","))
            pts.append((lo, la))
        else:
            name = s
    return polys


def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def footwall_nodes(faults, sub_dip=19.0):
    """SHELLS' Downer: footwall nodes of faults dipping at most sub_dip (dips < 0 mean 180 + dip), increasing order."""
    ch = set()
    for f in faults:
        a = [int(x) for x in f[1:5]]
        d = [float(x) + 180.0 if float(x) < 0.0 else float(x) for x in f[5:7]]
        for (near, far), dip in (((a[0], a[3]), d[0]), ((a[1], a[2]), d[1])):
            if near != far:
                if dip <= sub_dip:
                    ch.add(far)
                elif dip >= 180.0 - sub_dip:
                    ch.add(near)
    return sorted(ch)


def heal(inp, out):
    lines = open(f"{inp}/Earth5R.feg").read().splitlines()
    title, head = lines[0], lines[1].split()
    n = int(head[0])
    nodes = [l.split() for l in lines[2:2 + n]]
    ll = np.array([[float(f[1]), float(f[2])] for f in nodes])
    ne = int(lines[2 + n].split()[0])
    elems = [l.split() for l in lines[3 + n:3 + n + ne]]
    nf = int(lines[3 + n + ne].split()[0])
    faults = [l.split() for l in lines[4 + n + ne:4 + n + ne + nf]]
    assert all(int(f[0]) == k + 1 for k, f in enumerate(nodes)) and int(head[2]) == 0   # no fake nodes in Earth5R
    tri = np.array([[int(x) for x in e[1:4]] for e in elems]) - 1
    by_pair = {}
    for k, t in enumerate(tri):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            by_pair[frozenset((a, b))] = k
    polys = outlines(f"{inp}/PB2002_plates.dig")
    cent = [xyz(ll[t, 0], ll[t, 1]).sum(0) for t in tri]
    tplate = [next((nm for nm, g in polys if g.is_point_in_polygon(pygplates.PointOnSphere(c / np.linalg.norm(c)))), "")
              for c in cent]

    def plate(k):
        return tplate[k]

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    quads = [[int(x) - 1 for x in f[1:5]] for f in faults]
    heal_me = [plate(by_pair[frozenset((q[0], q[1]))]) == plate(by_pair[frozenset((q[2], q[3]))]) for q in quads]
    keep_idx = [k for k in range(nf) if not heal_me[k]]
    healed, refused = 0, 0
    for k in range(nf):
        if not heal_me[k]:
            continue
        q, saved = quads[k], parent[:]
        union(q[0], q[3]); union(q[1], q[2])
        if any(find(quads[j][0]) == find(quads[j][3]) and find(quads[j][1]) == find(quads[j][2]) for j in keep_idx):
            parent[:] = saved; heal_me[k] = False; keep_idx.append(k); refused += 1
        else:
            healed += 1
    kept = [faults[k] for k in sorted(keep_idx)]
    roots = sorted({find(i) for i in range(n)})
    new = {r: k + 1 for k, r in enumerate(roots)}
    nid = [new[find(i)] for i in range(n)]
    out_faults, dropped = [], 0
    for f in kept:
        q = [nid[int(x) - 1] for x in f[1:5]]
        if q[0] == q[3] and q[1] == q[2]:   # zero-DOF; should not happen after the check
            dropped += 1
            continue
        out_faults.append([str(len(out_faults) + 1)] + [str(x) for x in q] + f[5:])
    m = len(roots)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/Earth5R.feg", "w") as g:
        g.write(f"Earth5R.feg healed inside PB2002 plates (E-1119-16) from: {title[:40]}\n")
        g.write(f"{m:8d}{m:8d}{0:8d}{int(head[3]):8d}       T (numNod, nRealN, nFakeN, N1000, BRIEF)\n")
        for r in roots:
            g.write(f"{new[r]:8d} " + " ".join(nodes[r][1:]) + "\n")
        g.write(f"{ne:8d}\n")
        for k, t in enumerate(tri):
            g.write(f"{k + 1:8d}" + "".join(f"{nid[a]:8d}" for a in t) + (" " + " ".join(elems[k][4:]) if len(elems[k]) > 4 else "") + "\n")
        g.write(f"{len(out_faults):8d}\n")
        for f in out_faults:
            g.write(" ".join(f) + "\n")
    expected = footwall_nodes(out_faults)
    for b in ("Earth5R-type4A.bcs", "Earth5R-type4AplusA.bcs"):
        bl = open(f"{inp}/{b}").read().splitlines()
        cond, extra = {}, []
        for l in bl[1:]:
            f = l.split()
            if len(f) < 3:
                continue
            node = nid[int(f[1]) - 1]
            if f[2] == "5":
                extra.append((node, f[2:]))
            else:
                cond.setdefault(node, f[2:])
        assert all(e in cond for e in expected), "a footwall node of the healed grid had no condition"
        rows = [(e, cond[e]) for e in expected] + extra
        open(f"{out}/{b}", "w").write(bl[0] + "\n" + "\n".join(f"{k + 1:8d}{nd:8d} " + " ".join(r) for k, (nd, r) in enumerate(rows)) + "\n")
    return {"nodes": [n, m], "faults": [nf, len(out_faults)], "healed": healed, "refused": refused, "dropped_after_merge": dropped,
            "unassigned_triangles": tplate.count(""), "footwall_conditions": len(expected)}


if __name__ == "__main__":
    print(heal(sys.argv[1], sys.argv[2]))

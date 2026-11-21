"""Finite-element grid (.feg) for SHELLS made from plate-boundary lines alone (E-1119-16/-17), so that a generated planet
can be run the way the authors run Earth: the authors' Earth5R.feg was edited by hand in OrbWin, which a planet's many
candidates and time steps cannot be.

Grid, following the authors' grid as measured on Earth5R.feg (tools/diag, E-1119-16):
  - plate boundaries resampled at H_FAULT_KM (median length of the Earth5R.feg fault elements that lie on PB2002 plate
    boundaries, i.e. those kept by shells_feg_heal; all its faults, orogen faults included, have a median 84 km), junctions and the ends of
    every boundary segment kept exactly (also where a boundary ends on a vertex in the middle of another one)
  - the rest of the sphere filled with a Fibonacci lattice at H_PLATE_KM (median edge of Earth5R.feg farther than
    150 km from any fault; 231-235 km for cutoffs of 75-250 km); a lattice point is dropped if it lies inside the diametral circle of a boundary edge, which
    makes every boundary edge a Delaunay (Gabriel) edge; boundary edges still missing from the triangulation (where
    two boundaries meet at a sharp angle) are split at their midpoint and the triangulation is redone
  - triangulation: the convex hull of the unit vectors (= spherical Delaunay), corners turned counterclockwise
  - faults: along every boundary edge the nodes are split, one copy per sector of triangles between boundary edges
    (OrbWin's Cut); fault element order as SHELLS reads it (MOD_Data.f90, GetNet; checked on all 2846 faults of
    Earth5R.feg): the near triangle has the directed edge n2 -> n1 counterclockwise, the far one n4 -> n3
  - dips by boundary class (E-1119-16): spreading ridge and continental rift DIP_RIFT, subduction DIP_SUB toward the
    overriding plate, transform DIP_TRANSFORM, other convergent DIP_THRUST; a positive dip leans toward the n1, n2 side
  - nodal data: crust and mantle-lithosphere thickness are left 0 for OrbData to fill. Elevation and heat flow must be
    given: OrbData (OrbData5.f90) reads a heat-flow grid on unit 4 for any node with heat flow 0, and ShellSet supplies
    none (the authors' heat flows are stored in Earth5R.feg). For the Earth test, elevation is taken from the authors'
    INPUT/ETOPO20.grd with OrbData's own bilinear lookup (MOD_Data.f90, rows from the north), and heat flow from the
    nearest node of Earth5R.feg; a planet gives both from its history
  - boundary conditions: SHELLS lists the footwall nodes of faults dipping at most 19 degrees (Downer); one condition is
    written for each, in its order (shells_feg_heal.footwall_nodes)

Earth inputs (earth_boundaries): PB2002 boundaries (Bird 2003; title byte 3 "/" = the right-hand plate subducts under
the left-hand one, "\\" the reverse), step classes from PB2002_steps.dat. Not given by PB2002, and copied from the nearest
fault of Earth5R.feg for the Earth test only: which way the non-subduction faults dip (thrusts of continental
collision; ridges and rifts, whose two directions the authors use about equally); a planet takes them from its history.
Usage: .venv/bin/python -I tools/shells_grid.py <ShellSet INPUT dir> <PB2002_steps.dat> <out dir>"""
import os, sys, math
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

R_KM = 6371.0                   # Earth radius in iEarth5-049.in (m -> km)
H_FAULT_KM = 200.0              # median length of the Earth5R.feg fault elements on PB2002 plate boundaries (E-1119-17)
H_PLATE_KM = 235.0              # median edge length of Earth5R.feg farther than 150 km from faults (E-1119-17)
DIP_RIFT, DIP_SUB, DIP_TRANSFORM, DIP_THRUST = 55.0, 14.0, 90.0, 20.0   # degrees, by class (E-1119-16)
CLASS_DIP = {"OSR": DIP_RIFT, "CRB": DIP_RIFT, "SUB": DIP_SUB, "OTF": DIP_TRANSFORM, "CTF": DIP_TRANSFORM,
             "CCB": DIP_THRUST, "OCB": DIP_THRUST}
ARCHEAN_MA, PHANEROZOIC_MA = 2500.0, 541.0   # ICS chart; the age groups of E-1110-07
MW_PER_W = 1000.0
DIP_SIDE_KM = 100.0             # elevation of each side of a convergent fault sampled this far from it (E-1120-06)
HF14_FILE = "HFgrid14.grd"      # Lucazeau 2019 heat flow as the authors reformatted it (Earth5_update page, 2025)
HF14_MIN_W, HF14_MAX_W = 0.030, 0.300   # bounds the authors applied to HFgrid14 (same page), W/m^2
OROGENS = "PB2002_orogens.dig"  # Bird 2003 orogen outlines, shipped with the code (reference/earth/plate-boundaries/)
OROGEN_CFRIC = 0.205            # continuum friction inside weak orogens: the authors' 2026 Earth model (E-1120-07)
FAULT_LR0 = 2                   # technical: first rheology number of the per-class fault frictions (1 = weak orogens)
SNAP_KM = 1.0                   # PB2002 line ends meant to meet: <= 0.26 km apart; distinct junctions >= 28.8 km (E-1119-17)


def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def lonlat(p):
    p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(p[..., 1], p[..., 0])), np.degrees(np.arcsin(np.clip(p[..., 2], -1, 1)))


def slerp_line(pts, h):
    """points along a polyline (unit vectors) at spacing <= h (radians), ends kept."""
    seg = np.arccos(np.clip((pts[:-1] * pts[1:]).sum(1), -1, 1))
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(1, math.ceil(s[-1] / h))
    out = []
    for t in np.linspace(0, s[-1], n + 1):
        k = min(np.searchsorted(s, t, side="right") - 1, len(seg) - 1)
        f = 0.0 if seg[k] == 0 else (t - s[k]) / seg[k]
        p = pts[k] * (1 - f) + pts[k + 1] * f
        out.append(p / np.linalg.norm(p))
    return np.array(out)


def fibonacci(n):
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = k * math.pi * (3 - math.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def build(lines, h_fault=H_FAULT_KM / R_KM, h_plate=H_PLATE_KM / R_KM):
    """lines: list of (polyline unit vectors, info dict). Returns nodes (n, 3), triangles (m, 3) counterclockwise,
    boundary edges {frozenset(a, b): info} on the unsplit nodes."""
    key = lambda p: tuple(np.round(p, 9))
    index, nodes = {}, []

    def node(p):
        k = key(p)
        if k not in index:
            index[k] = len(nodes); nodes.append(p)
        return index[k]

    # a boundary may end on a vertex in the middle of another one (T junction): that vertex must stay a node, so each
    # line is cut at its vertices where another line ends before resampling
    ends = {key(pts[0]) for pts, _ in lines} | {key(pts[-1]) for pts, _ in lines}
    bedges = {}
    for pts, info in lines:
        cuts = [0] + [k for k in range(1, len(pts) - 1) if key(pts[k]) in ends] + [len(pts) - 1]
        for i0, i1 in zip(cuts[:-1], cuts[1:]):
            q = slerp_line(pts[i0:i1 + 1], h_fault)
            ids = [node(p) for p in q]
            for a, b in zip(ids[:-1], ids[1:]):
                if a != b:
                    bedges[frozenset((a, b))] = info
    nb = len(nodes)
    # Fibonacci lattice: area per node of an equilateral mesh of edge h is (sqrt(3)/2) h^2
    seeds = fibonacci(int(round(4 * math.pi / (math.sqrt(3) / 2 * h_plate ** 2))))
    P = np.array(nodes)
    for _ in range(30):
        E = np.array([tuple(e) for e in bedges])
        mid = P[E[:, 0]] + P[E[:, 1]]
        half = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1) / 2          # chord half-length
        mid /= np.linalg.norm(mid, axis=1)[:, None]
        # the ends of each edge too, with the same radius: a lattice point next to a boundary node would otherwise
        # make a sliver triangle (outside every diametral circle, which only touch the ends)
        mid, half = np.concatenate([mid, P[E[:, 0]], P[E[:, 1]]]), np.concatenate([half, half, half])
        tree = cKDTree(mid)
        keep = np.ones(len(seeds), bool)
        for i, js in enumerate(tree.query_ball_point(seeds, half.max())):
            for j in js:
                if np.linalg.norm(seeds[i] - mid[j]) < half[j]:
                    keep[i] = False; break
        allp = np.concatenate([P, seeds[keep]])
        hull = ConvexHull(allp)
        tri = hull.simplices.copy()
        flip = np.einsum("ij,ij->i", np.cross(allp[tri[:, 1]] - allp[tri[:, 0]], allp[tri[:, 2]] - allp[tri[:, 0]]), allp[tri[:, 0]]) < 0
        tri[flip] = tri[flip][:, ::-1]
        have = set()
        for t in tri:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                have.add(frozenset((a, b)))
        missing = [e for e in bedges if e not in have]
        if not missing:
            return allp, tri, bedges, nb
        new = []
        for e in missing:       # split at the midpoint, keep the boundary's info
            a, b = tuple(e)
            m = P[a] + P[b]; m /= np.linalg.norm(m)
            info = bedges.pop(e)
            c = len(P) + len(new); new.append(m)
            bedges[frozenset((a, c))] = info; bedges[frozenset((c, b))] = info
        P = np.vstack([P, np.array(new)])
        nb = len(P)
        print(f"  round {_}: {len(missing)} boundary edges missing, split", flush=True)
    raise RuntimeError("boundary edges still missing after 30 splitting rounds")


def cut_faults(P, tri, bedges):
    """split nodes along boundary edges; returns new P, tri, and fault quads [(n1, n2, n3, n4, info, near_tri)]."""
    tri = tri.copy()
    around = {}
    for k, t in enumerate(tri):
        for j in range(3):
            around.setdefault(t[j], []).append(k)
    bnodes = {v for e in bedges for v in e}
    P = list(P)
    for v in bnodes:
        ts = around[v]
        par = {k: k for k in ts}

        def f(k):
            while par[k] != k:
                k = par[k]
            return k
        byedge = {}
        for k in ts:
            for u in tri[k]:
                if u != v:
                    byedge.setdefault(u, []).append(k)
        for u, ks in byedge.items():
            if len(ks) == 2 and frozenset((u, v)) not in bedges:
                par[f(ks[0])] = f(ks[1])
        sectors = sorted({f(k) for k in ts})
        for s in sectors[1:]:
            c = len(P); P.append(P[v])
            for k in ts:
                if f(k) == s:
                    tri[k][list(tri[k]).index(v)] = c
    P = np.array(P)
    # the unsplit id of every node (the copies were appended, so the first id at a place), to find the boundary edge
    # each triangle side lies on
    pos = {}
    for i, p in enumerate(P):
        pos.setdefault(tuple(np.round(p, 12)), i)
    o = np.array([pos[tuple(np.round(p, 12))] for p in P])
    sides = {}
    for k, t in enumerate(tri):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e = frozenset((o[a], o[b]))
            if e in bedges:
                sides.setdefault(e, []).append((k, a, b))
    quads = []
    for e, ss in sides.items():
        assert len(ss) == 2, "a boundary edge is not between two triangles"
        (kn, x, y), (kf, x2, y2) = ss          # near: x -> y counterclockwise, so n2 = x, n1 = y
        assert o[x2] == o[y] and o[y2] == o[x]
        quads.append((y, x, y2, x2, bedges[e], kn, kf))   # far: x2 -> y2 = (copy of y) -> (copy of x): n4 = x2, n3 = y2
    return P, tri, quads


def write_feg(fn, title, P, tri, faults, elev=None, heat=None, data=None, lr=None):
    """data: the other four nodal columns (orbdata_nodes.assign_nodes: zM, tL, chem_drho, cooling_curv); 0 = left for
    ShellSet's OrbData to fill; lr: per triangle, True = lithospheric rheology 1 (SHELLS_v5.0 "LR" flag); faults:
    (nodes, dip1, dip2, LR#), LR# 0 = the default rheology"""
    lon, lat = lonlat(P)
    elev = np.zeros(len(P)) if elev is None else elev
    heat = np.zeros(len(P)) if heat is None else heat
    rest = np.zeros((len(P), 4)) if data is None else np.stack([data[k] for k in ("zM", "tL", "chem_drho", "cooling_curv")], 1)
    with open(fn, "w") as g:
        g.write(title[:80] + "\n")
        g.write(f"{len(P):8d}{len(P):8d}{0:8d}{max(100000, 10 * len(P)):8d}       T (numNod, nRealN, nFakeN, N1000, BRIEF)\n")
        for i in range(len(P)):
            g.write(f"{i + 1:8d} {lon[i]:10.5f} {lat[i]:10.5f} {elev[i]:.3E} {heat[i]:.3E} " + " ".join(f"{x:.2E}" for x in rest[i]) + "\n")
        g.write(f"{len(tri):8d} (numEl = NUMBER OF TRIANGULAR CONTINUUM ELEMENTS)\n")
        for k, t in enumerate(tri):
            g.write(f"{k + 1:8d}{t[0] + 1:8d}{t[1] + 1:8d}{t[2] + 1:8d}" + (" LR1" if lr is not None and lr[k] else "") + "\n")
        g.write(f"{len(faults):8d} (nFl =  NUMBER OF CURVILINEAR FAULT ELEMENTS)\n")
        for k, (q, d1, d2, n) in enumerate(faults):
            g.write(f"{k + 1:8d}" + "".join(f"{n + 1:8d}" for n in q) + f" {d1:5.1f} {d2:5.1f} 0.00E+00" + (f" LR{n}" if n else "") + "\n")


def half_bandwidth(n, tri, quads):
    """SHELLS' KSize (MOD_Shells.f90): the largest node-number distance between two nodes of one triangle or one fault"""
    lo, hi = np.arange(n), np.arange(n)
    for el in [np.asarray(tri)] + ([np.asarray(quads)] if len(quads) else []):
        for j in range(el.shape[1]):
            np.minimum.at(lo, el[:, j], el.min(1)); np.maximum.at(hi, el[:, j], el.max(1))
    return int(max((np.arange(n) - lo).max(), (hi - np.arange(n)).max()))


def renumber(P, tri, quads):
    """node order of small bandwidth, as the authors' grids get from OrbNumber (OrbScore2.f90 header): SHELLS stores the
    stiffness matrix as a band and solves it with LAPACK dgbsv, so memory and time grow with the half-bandwidth.
    Reverse Cuthill-McKee (Cuthill and McKee 1969; George 1971) on the graph of nodes sharing a triangle or a fault
    element, the same couplings KSize uses. Returns P, tri, quads renumbered."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    els = [np.asarray(tri)] + ([np.asarray(quads)] if len(quads) else [])
    r = np.concatenate([np.repeat(e, e.shape[1], 1).ravel() for e in els])
    c = np.concatenate([np.tile(e, e.shape[1]).ravel() for e in els])
    perm = reverse_cuthill_mckee(coo_matrix((np.ones(len(r)), (r, c)), shape=(len(P), len(P))).tocsr(), symmetric_mode=True)
    new = np.empty(len(P), int); new[perm] = np.arange(len(P))
    return P[perm], new[np.asarray(tri)], new[np.asarray(quads)]


# ---------------------------------------------------------------- Earth (PB2002) inputs
def grd_lookup(fn, lon, lat):
    """OrbData's bilinear lookup in one of Bird's .grd files (MOD_Data.f90, elevation branch): line 1 x1 dx x2, line 2
    y1 dy y2, then rows from y2 (north) southward."""
    tok = open(fn).read().split()
    x1, dx, x2, y1, dy, y2 = (float(t) for t in tok[:6])
    nx, ny = int((x2 - x1) / dx + 1.5), int((y2 - y1) / dy + 1.5)
    a = np.array(tok[6:6 + nx * ny], float).reshape(ny, nx)
    lon = np.where(lon < x1, lon + 360.0, np.where(lon > x2, lon - 360.0, lon))
    ir1 = np.clip(((y2 - lat) / dy + 1.00001).astype(int), 1, ny - 1)
    fr = np.clip(((y2 - dy * (ir1 - 1)) - lat) / dy, 0, 1)
    ic1 = np.clip(((lon - x1) / dx + 1.00001).astype(int), 1, nx - 1)
    fc = np.clip((lon - (x1 + dx * (ic1 - 1))) / dx, 0, 1)
    r1, c1 = ir1 - 1, ic1 - 1
    top = a[r1, c1] + fc * (a[r1, c1 + 1] - a[r1, c1])
    bot = a[r1 + 1, c1] + fc * (a[r1 + 1, c1 + 1] - a[r1 + 1, c1])
    return top + fr * (bot - top)


def read_dig(fn):
    out, title, pts = [], None, []
    for l in open(fn):
        s = l.rstrip("\n")
        if not s.strip():
            continue
        if s.strip().startswith("***"):
            out.append((title, xyz(*np.array(pts).T))); pts = []
        elif s.strip()[0] in "+-0123456789" and "," in s:
            pts.append([float(x) for x in s.split(",")])
        else:
            title = s
    return out


def outlines(fn):
    import pygplates
    return [(t.strip()[:2], pygplates.PolygonOnSphere([tuple(lonlat(p)[::-1]) for p in pts])) for t, pts in read_dig(fn)]


def plate_of(polys, p):
    import pygplates
    q = pygplates.PointOnSphere(*[float(x) for x in lonlat(p)[::-1]])
    return next((nm for nm, g in polys if g.is_point_in_polygon(q)), "")


def element_classes(P, ends, stree, scls, slen):
    """class of each fault element (node pairs ends): the class covering most of its length, each boundary step going to
    the element nearest to its midpoint (point-to-segment distance among the elements with the 8 nearest midpoints).
    A 200 km element spans several steps of a ridge-transform staircase; the class of the step nearest to its midpoint
    made 79 of 267 vertical elements of the Earth test cover mostly ridge, against 11 of 151 in Earth5R.feg, and locked
    those ridges against opening (E-1120-03). Elements that get no step keep the class of the nearest step."""
    A, B = P[ends[:, 0]], P[ends[:, 1]]
    mid = (A + B) / np.linalg.norm(A + B, axis=1)[:, None]
    S = stree.data
    _, cand = cKDTree(mid).query(S, k=min(8, len(mid)))
    a, b = A[cand], B[cand]
    t = np.clip(np.einsum("skx,skx->sk", S[:, None] - a, b - a) / np.einsum("skx,skx->sk", b - a, b - a), 0, 1)
    d = np.linalg.norm(S[:, None] - (a + t[..., None] * (b - a)), axis=2)
    own = cand[np.arange(len(S)), d.argmin(1)]
    share = {}
    for e, c, w in zip(own, scls, slen):
        share.setdefault(e, {}).setdefault(c, 0.0)
        share[e][c] += w
    near = stree.query(mid)[1]
    return [max(share[e], key=share[e].get) if e in share else scls[near[e]] for e in range(len(ends))]


def earth_boundaries(inp, steps_fn):
    steps = [l.replace(":", " ").split() for l in open(steps_fn) if l.strip()]
    sa = xyz(np.array([float(s[2]) for s in steps]), np.array([float(s[3]) for s in steps]))
    sb = xyz(np.array([float(s[4]) for s in steps]), np.array([float(s[5]) for s in steps]))
    smid = (sa + sb) / np.linalg.norm(sa + sb, axis=1)[:, None]
    stree, scls = cKDTree(smid), [s[-1].rstrip("*") for s in steps]
    slen = np.linalg.norm(sa - sb, axis=1)
    lines = []
    dig = read_dig(f"{inp}/PB2002_boundaries.dig")
    # PB2002 digitizing: at three junctions the line ends meant to meet are 0.16-0.26 km apart; each vertex within
    # SNAP_KM of a line end is moved onto it (lower-numbered end first), so that junction is one node
    ends = np.array([p[k] for _, p in dig for k in (0, -1)])
    rep = np.arange(len(ends))
    for i, j in sorted(cKDTree(ends).query_pairs(SNAP_KM / R_KM)):
        rep[j] = rep[i] if rep[j] == j else rep[j]
    ends = ends[rep]
    etree = cKDTree(ends)
    for _, pts in dig:
        d, k = etree.query(pts)
        near = d < SNAP_KM / R_KM
        pts[near] = ends[k[near]]
    for title, pts in dig:
        pair = title.strip()[:5]
        lines.append((pts, {"left": pair[:2], "right": pair[3:5], "sym": pair[2]}))
    return lines, stree, scls, slen


def heat_from_history(ocean_age_ma, cont_age_ma, groups):
    """node heat flow (W/m^2) from what a planet's history gives (E-1120-05): sea floor (age < AGE_MAX_MA, OrbData's own
    ocean test) by GDH1 of Stein & Stein 1992 at its age, as OrbData does for oceanic lithosphere and as the authors'
    Earth5R.feg has it (zero age: OrbData's upper limit qLim1); continental crust by the median measured heat flow of its
    formation-age group (E-1110-07; nan = outside the craton polygons). groups: tools/results/earth_heatflow_by_age.json"""
    import orbdata_nodes as od
    q = np.empty(len(ocean_age_ma))
    for i, (a, c) in enumerate(zip(ocean_age_ma, cont_age_ma)):
        if a < od.AGE_MAX_MA:
            q[i] = od.QLIM1 if a <= 0 else min(od._stein(a), od.QLIM1)
        else:
            k = ("outside the craton polygons" if np.isnan(c) else "Archean (>= 2500 Ma)" if c >= ARCHEAN_MA else
                 "Proterozoic (541-2500 Ma)" if c >= PHANEROZOIC_MA else "craton polygon younger than 541 Ma")
            q[i] = groups[k]["median"] / MW_PER_W
    return q


def earth_heat_rule(inp, lon, lat):
    """Earth inputs of heat_from_history: sea-floor age from the authors' INPUT/age_1p5.grd (OrbData's lookup), formation
    age of the Müller 2022 craton polygon holding the node (the oldest, as E-1110-07)"""
    import json, pygplates, orbdata_nodes as od
    from plate_checks import MODEL, HERE
    age = od.Grid(f"{inp}/{od.GRID_FILES['age']}")
    a = np.array([age.lookup(x, y) for x, y in zip(lon, lat)])
    polys = [(g, f.get_valid_time()[0]) for f in pygplates.FeatureCollection(str(MODEL / "Cratons/cratons.gpml"))
             for g in f.get_geometries() if isinstance(g, pygplates.PolygonOnSphere)]
    c = np.full(len(lon), np.nan)
    for i in np.where(a >= od.AGE_MAX_MA)[0]:
        p = pygplates.PointOnSphere(lat[i], lon[i])
        c[i] = max([t for g, t in polys if g.is_point_in_polygon(p)], default=np.nan)
    return heat_from_history(a, c, json.loads((HERE / "results/earth_heatflow_by_age.json").read_text()))


def toward_higher(m, near, zfun):
    """+1 if the fault through m dips toward its near side (unit normal near), else -1, by the rule of E-1120-06: a
    thrust dips under the higher side, i.e. under the hinterland of the orogen (the former upper plate of a collision,
    the arc side of a retro-arc belt), sampled DIP_SIDE_KM away on each side; zfun(lon, lat) gives the elevation"""
    a = DIP_SIDE_KM / R_KM
    lon, lat = lonlat(np.array([np.cos(a) * m + np.sin(a) * near, np.cos(a) * m - np.sin(a) * near]))
    zn, zf = zfun(lon, lat)
    return 1.0 if zn > zf else -1.0


def weak_orogens(P, tri, dig_fn, names):
    """triangles whose centroid lies inside the named orogen outlines ("all": every one), as the authors' Assign_LRs.f90
    assigns their weak rheology (Earth5-2023d.feg: Persia-Tibet-Burma only)"""
    import pygplates
    polys = [pygplates.PolygonOnSphere([tuple(lonlat(p)[::-1]) for p in pts]) for t, pts in read_dig(dig_fn)
             if names == "all" or t.split("  ")[0].strip() in names]
    c = P[np.asarray(tri)].sum(1)
    lon, lat = lonlat(c / np.linalg.norm(c, axis=1)[:, None])
    return np.array([any(g.is_point_in_polygon(pygplates.PointOnSphere(float(y), float(x))) for g in polys)
                     for x, y in zip(lon, lat)])


def write_lr(in_fn, out_fn, cfric=None, fault_fric=()):
    """SHELLS_v5.0 table of lithospheric rheologies, each the default rheology of the parameter file in_fn with one value
    changed (columns as in the authors' LR_set_for_Earth5-2023d.dat): LR1 continuum friction cfric (weak orogens, if
    given), then LR2, LR3, ... fault friction of each (class, fFric) of fault_fric"""
    v = [l.split()[0] for l in open(in_fn).read().splitlines()[1:10]]
    row = lambda ff, cf: " ".join([str(ff), str(cf), v[2], v[3]] + [x for vv in v[4:8] for x in vv.split(",")] + [v[8]])
    rows = ([f"1 {row(v[0], cfric)}"] if cfric is not None else []) + \
           [f"{FAULT_LR0 + k} {row(ff, v[1])}" for k, (_, ff) in enumerate(fault_fric)]
    open(out_fn, "w").write("LR# fFric cFric Biot Byerly aCreep(1) aCreep(2) bCreep(1) bCreep(2) cCreep(1) cCreep(2) "
                            "dCreep(1) dCreep(2) eCreep\n" + "\n".join(rows) + "\n")


def earth_grid(inp, steps_fn, out, refine=1.0, own_data=False, heat_rule=False, flip=(), dip_elev=False, weak=None,
               fault_fric=()):
    """refine: both spacings divided by it (grid convergence check); own_data: nodal data from orbdata_nodes (the
    port of OrbData, E-1119-17) instead of leaving them to ShellSet's OrbData; heat_rule: node heat flow from
    heat_from_history instead of the authors' nodes ("hf14": from HFgrid14, the authors' 2026 choice, control only); flip: boundary classes whose dip direction is reversed (sensitivity
    test of the copied directions, E-1120-05); dip_elev: continental and ocean-continent convergent faults dip toward
    the higher side (toward_higher, E-1120-06) instead of copying the authors' directions; weak: orogen names (or "all")
    whose triangles get continuum friction OROGEN_CFRIC (rheology 1, file LR_set.dat; E-1120-07); fault_fric: (class,
    fFric) pairs, faults of these classes get their own friction (rheologies 2, 3, ...; E-1121-03), the others keep the
    default fFric"""
    lines, stree, scls, slen = earth_boundaries(inp, steps_fn)
    P, tri, bedges, nb = build(lines, H_FAULT_KM / refine / R_KM, H_PLATE_KM / refine / R_KM)
    P, tri, quads = cut_faults(P, tri, bedges)
    polys = outlines(f"{inp}/PB2002_plates.dig")
    # authors' faults, for the dip direction of non-subduction faults (Earth test only)
    L = open(f"{inp}/Earth5R.feg").read().splitlines()
    n0 = int(L[1].split()[0]); ne0 = int(L[2 + n0].split()[0])
    Q = xyz(*np.array([[float(x) for x in l.split()[1:3]] for l in L[2:2 + n0]]).T)
    F0 = [l.split() for l in L[4 + n0 + ne0:] if len(l.split()) >= 7]
    fmid = np.array([Q[int(f[1]) - 1] + Q[int(f[2]) - 1] for f in F0]); fmid /= np.linalg.norm(fmid, axis=1)[:, None]
    # unit normal of each authors' fault pointing to its near side (n1, n2): (n1 x n2) points left of n1 -> n2, and
    # the near triangle holds n2 -> n1 counterclockwise, i.e. lies to the right of n1 -> n2
    fnear = -np.cross(Q[[int(f[1]) - 1 for f in F0]], Q[[int(f[2]) - 1 for f in F0]])
    fnear /= np.linalg.norm(fnear, axis=1)[:, None]
    fsign = np.array([1.0 if float(f[5]) >= 0 else -1.0 for f in F0])
    ftree = cKDTree(fmid)
    faults, count = [], {}
    ecls = element_classes(P, np.array([qd[:2] for qd in quads]), stree, scls, slen)
    for (n1, n2, n3, n4, info, kn, kf), cls in zip(quads, ecls):
        m = P[n1] + P[n2]; m /= np.linalg.norm(m)
        dip = CLASS_DIP[cls]
        if cls == "SUB" and info["sym"] in "/\\":
            over = info["left"] if info["sym"] == "/" else info["right"]
            cn = P[tri[kn]].sum(0)
            sign = 1.0 if plate_of(polys, cn) == over else -1.0
        elif dip == DIP_TRANSFORM:
            sign = 1.0
        elif dip_elev and cls in ("CCB", "OCB"):
            near = -np.cross(P[n1], P[n2]); near /= np.linalg.norm(near)
            sign = toward_higher(m, near, lambda x, y: grd_lookup(f"{inp}/ETOPO20.grd", x, y))
        else:
            j = ftree.query(m)[1]
            near = -np.cross(P[n1], P[n2]); near /= np.linalg.norm(near)
            sign = fsign[j] * (1.0 if near @ fnear[j] > 0 else -1.0)
        if cls in flip:
            sign = -sign
        lrs = [c for c, _ in fault_fric]
        faults.append(((n1, n2, n3, n4), sign * dip, sign * dip, FAULT_LR0 + lrs.index(cls) if cls in lrs else 0))
        count[cls] = count.get(cls, 0) + 1
    q = np.array([f[0] for f in faults])
    bw0 = half_bandwidth(len(P), tri, q)
    P, tri, q = renumber(P, tri, q)
    faults = [(tuple(int(x) for x in qq), d1, d2, n) for qq, (_, d1, d2, n) in zip(q, faults)]
    bw = half_bandwidth(len(P), tri, q)
    os.makedirs(out, exist_ok=True)
    lon, lat = lonlat(P)
    elev = grd_lookup(f"{inp}/ETOPO20.grd", lon, lat)
    if heat_rule == "hf14":     # control only (E-1120-06): the authors' 2026 heat flow, with their bounds
        import orbdata_nodes as od
        g = od.Grid(HF14_FILE)
        heat = np.clip([g.lookup(x, y) for x, y in zip(lon, lat)], HF14_MIN_W, HF14_MAX_W)
    elif heat_rule:
        heat = earth_heat_rule(inp, lon, lat)
    else:
        heat = np.array([float(l.split()[4]) for l in L[2:2 + n0]])[cKDTree(Q).query(P)[1]]
    data = None
    if own_data:
        import orbdata_nodes
        data = orbdata_nodes.assign_nodes(lon, lat, elev, heat, inp)
        elev, heat = data["elev"], data["heat"]
    lr = None
    if weak:
        lr = weak_orogens(P, tri, os.path.join(os.path.dirname(steps_fn), OROGENS), weak)
    if weak or fault_fric:
        write_lr(f"{inp}/iEarth5-049.in", f"{out}/LR_set.dat", OROGEN_CFRIC if weak else None, fault_fric)
    write_feg(f"{out}/Earth5R.feg", "Generated from PB2002 boundaries (tools/shells_grid.py, E-1119-17)", P, tri, faults,
              elev, heat, data, lr)
    # boundary conditions: footwall nodes in SHELLS' order; condition and plate-interior nodes from the authors' files
    import shells_feg_heal as fh
    rows = [f.split() for f in open(f"{out}/Earth5R.feg").read().splitlines()[4 + len(P) + len(tri):]]
    expected = fh.footwall_nodes([r for r in rows if len(r) >= 7])
    ptree = cKDTree(P)
    # a plate-interior condition goes to the nearest node that is inside the named plate and not on a fault: the split
    # nodes of a generated fault lie exactly on the PB2002 line, so the nearest node alone may be a fault node (E-1120-04)
    on_fault = {int(x) for r in rows if len(r) >= 7 for x in r[1:5]}
    for b in ("Earth5R-type4A.bcs", "Earth5R-type4AplusA.bcs"):
        bl = open(f"{inp}/{b}").read().splitlines()
        body, used = [(e, ["4"]) for e in expected], set(expected)
        for l in bl[1:]:
            f = l.split()
            if len(f) >= 4 and f[2] == "5":
                order = ptree.query(Q[int(f[1]) - 1], k=len(P))[1] + 1
                nd = next((int(c) for c in order if c not in on_fault and plate_of(polys, P[c - 1]) == f[3].strip('"')),
                          int(order[0]))
                if nd not in used:
                    used.add(nd); body.append((nd, f[2:]))
        open(f"{out}/{b}", "w").write(bl[0] + " [regenerated, E-1119-17]\n" + "\n".join(
            f"{k + 1:8d}{nd:8d} " + " ".join(r) for k, (nd, r) in enumerate(body)) + "\n")
    return {"nodes": len(P), "boundary_nodes_unsplit": nb, "triangles": len(tri), "faults": len(faults),
            "classes": count, "footwall_conditions": len(expected), "half_bandwidth_before_after": [bw0, bw],
            "weak_triangles": int(lr.sum()) if lr is not None else 0}


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print(earth_grid(sys.argv[1], sys.argv[2], sys.argv[3], *(float(a) for a in sys.argv[4:5]), own_data=len(sys.argv) > 5))

"""Material-tracking plate engine (RULES.md A2, A4, A10, A11).

Crust is carried by tracer particles that move continuously with their plate's rotation (pyGPlates), the same idea as
GPlately's seafloor age algorithm. A fixed check grid covers the sphere. Each step (dt Myr towards the present):
  1. every plate's particles are rotated by that plate's stage rotation;
  2. a grid point with no particle nearby lies in a gap between diverging plates -> a new oceanic particle is born there
     (age 0), belonging to the nearest plate (symmetric spreading);
  3. where particles of different plates crowd the same grid point (a convergent boundary), continental crust stays and
     the younger ocean stays; the older oceanic particles are consumed (subduction). Continental particles are never
     deleted (RULES.md A10).
The state on the grid (each grid point takes its nearest particle) therefore has exactly one plate everywhere: no gaps,
no overlaps (A2).

All lengths use the radius passed in (never a tool's Earth default, RULES.md E9).
"""
import math
import numpy as np
from fastkd import cKDTree

# how a particle became continental crust (record "cont_src" when the caller keeps it, with "cont_since" the time;
# 2026-11-13, EVIDENCE E-1112-14): labels only, used by no rule (0 = ocean)
CONT_SRC_START, CONT_SRC_RIFT, CONT_SRC_SEDIMENT, CONT_SRC_POCKET = 1, 2, 3, 4


def fibonacci_xyz(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    th = math.pi * (3 - math.sqrt(5)) * np.arange(n)
    return np.column_stack([r * np.cos(th), r * np.sin(th), z])


def xyz_to_latlon(xyz):
    return np.degrees(np.arcsin(np.clip(xyz[:, 2], -1, 1))), np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))


def rotation_matrix(fr):
    """3x3 matrix of a pygplates.FiniteRotation."""
    if fr.represents_identity_rotation():
        return np.eye(3)
    lat, lon, ang = fr.get_lat_lon_euler_pole_and_angle_degrees()
    la, lo, a = math.radians(lat), math.radians(lon), math.radians(ang)
    k = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


class PlateEngine:
    def __init__(self, n_grid, radius_km, stage_rotation, gap_r=1.0, overlap_r=0.5):
        """stage_rotation(plate_id, t_from, t_to) -> pygplates.FiniteRotation moving plate material from t_from to t_to."""
        self.grid = fibonacci_xyz(n_grid)
        self.grid_tree = cKDTree(self.grid)
        self.radius_km = radius_km
        self.spacing = math.sqrt(4 * math.pi / n_grid)            # mean grid spacing on the unit sphere (radians)
        self.cell_area_km2 = 4 * math.pi * radius_km ** 2 / n_grid
        self.stage_rotation = stage_rotation
        self.gap_r, self.overlap_r = gap_r, overlap_r  # in grid spacings
        self.log = []
        # particles start on the grid points; callers fill plate / cont / birth / block
        self.pos = self.grid.copy()
        self.plate = np.zeros(n_grid, np.int32)
        self.cont = np.zeros(n_grid, bool)
        self.birth = np.full(n_grid, np.nan)          # time the oceanic crust formed (Ma); NaN = unknown or continental
        self.block = np.full(n_grid, -1, np.int32)    # continental block id (RULES.md C1)
        # extra per-particle records carried with the crust (e.g. when and by which collision it was thickened);
        # name -> (array, value given to newly made ocean crust)
        self.extra = {}
        # plate pairs in their continental rift stage: gaps between them are stretched continental crust, not ocean
        self.rift_pairs = set()
        # continental collision with volume conservation (2026-11-05): a continental particle overlapped by continental
        # crust of another plate is consumed like ocean, and its crust is handed to the particle that overrode it (the
        # caller thickens the belt from last_cont_consumed); off = continents pile up unseen (old behaviour)
        self.cont_collide = False
        self.last_cont_consumed = None

    def add_record(self, name, values, new_value):
        self.extra[name] = (np.asarray(values).copy(), new_value)

    def _priority(self, idx):
        # continental crust first, then younger ocean (later birth = smaller Ma), unknown-age ocean counts as oldest
        return np.where(self.cont[idx], 1e9, -np.nan_to_num(self.birth[idx], nan=1e6))

    def step(self, t_old, t_new, assign=None):
        """assign(t, positions) -> plate ids: optional plate membership from outside (e.g. an Earth model's topologies,
        or our planet's plate reorganisation events). Without it, particles keep their plate."""
        self._gj = None                                       # grid -> nearest particle cache (see grid_state)
        if assign is not None:
            self.plate = np.asarray(assign(t_old, self.pos), dtype=np.int32)
        # 1. move every plate's particles
        for p in np.unique(self.plate):
            sel = self.plate == p
            M = rotation_matrix(self.stage_rotation(int(p), t_old, t_new))
            self.pos[sel] = self.pos[sel] @ M.T
        self.pos /= np.linalg.norm(self.pos, axis=1, keepdims=True)
        tree = cKDTree(self.pos)

        # 2. gaps: grid points with no particle within gap_r -> new ocean floor
        d, nearest = tree.query(self.grid)
        gap = d > self.gap_r * self.spacing
        n_new = int(gap.sum())

        # 3. overlaps: a particle closer than overlap_r to a particle of another plate with higher priority is consumed
        dist, nb = tree.query(self.pos, k=8, distance_upper_bound=self.overlap_r * self.spacing)
        valid = nb < len(self.pos)
        nb_safe = np.where(valid, nb, 0)
        pri = self._priority(np.arange(len(self.pos)))
        pn, pp = pri[nb_safe], pri[:, None]
        # a tie (e.g. two oceans of unknown age) is broken by plate id, so one boundary keeps one subduction polarity
        stronger = (pn > pp) | ((pn == pp) & (self.plate[nb_safe] > self.plate[:, None]))
        tp_ = getattr(self, "trench_pairs", None)
        if tp_:
            # at a trench the subducting plate is consumed, not the overriding one (the trench polarity of the force model,
            # plate_forces.trenches: (sinker, overrider)); elsewhere the rule above. ENGINE_POLARITY=1 in planet_m1.
            pa_, pb_ = self.plate[:, None], self.plate[nb_safe]
            key_ = lambda a, b: a.astype(np.int64) * 1000003 + b
            tk_ = np.array([s_ * 1000003 + o_ for s_, o_ in tp_], np.int64)
            sinks_ = np.isin(key_(pa_, pb_), tk_); overs_ = np.isin(key_(pb_, pa_), tk_)
            oo_ = ~self.cont[:, None] & ~self.cont[nb_safe]
            stronger = np.where(oo_ & sinks_, True, np.where(oo_ & overs_, False, stronger))
        other = valid & (self.plate[nb_safe] != self.plate[:, None]) & stronger
        consumed = (~self.cont) & other.any(axis=1)
        self.last_cont_consumed = None
        if self.cont_collide:
            oc = other & self.cont[nb_safe] & self.cont[:, None]
            cc_ = self.cont & oc.any(axis=1)
            if cc_.any():
                src = np.nonzero(cc_)[0]
                rcp = nb_safe[src, np.argmax(oc[src], axis=1)]          # the nearest continental particle that overrode it
                ok = ~cc_[rcp]                                         # (a receiver consumed in turn: hand on is skipped)
                consumed |= cc_
                newidx = np.cumsum(~consumed) - 1
                self.last_cont_consumed = {"receiver": newidx[rcp[ok]], "pos": self.pos[src[ok]].copy(),
                                           "plates": np.c_[self.plate[src[ok]], self.plate[rcp[ok]]],
                                           "extra": {k: a[src[ok]].copy() for k, (a, _) in self.extra.items()},
                                           "lost": int((~ok).sum())}
        n_consumed = int(consumed.sum())
        oc_ = np.nonzero(consumed & ~self.cont)[0]              # diagnostic: (consumed plate, plate that overrode it)
        self.last_consumed_pairs = np.c_[self.plate[oc_], self.plate[nb_safe[oc_, np.argmax(other[oc_], axis=1)]]]
        # consumed ocean particles' position and records, for a caller that keeps a quantity they carry (planet_m1: margin
        # sediment waiting to become crust, conserve_orogen_volume)
        self.last_ocean_consumed = {"pos": self.pos[oc_].copy(), "extra": {k: a[oc_].copy() for k, (a, _) in self.extra.items()}}

        keep = ~consumed
        new_pos = self.grid[gap]
        new_plate = self.plate[nearest[gap]]                    # symmetric spreading: the nearest plate takes it
        new_cont = np.zeros(n_new, bool)
        if self.rift_pairs and n_new:
            # during a rift stage the opening gap is filled by stretched continental crust
            dk, kk = tree.query(self.grid[gap], k=12)
            pl = self.plate[kk]
            first = pl[:, 0]
            other = np.where(pl != first[:, None], pl, -1)
            second = np.array([o[o >= 0][0] if (o >= 0).any() else -1 for o in other])
            both_cont = self.cont[kk[:, 0]] & np.array([self.cont[kk[r][pl[r] == second[r]][0]] if second[r] >= 0 else False
                                                        for r in range(n_new)])
            in_rift = np.array([(int(a), int(b)) in self.rift_pairs for a, b in zip(first, second)])
            new_cont = both_cont & in_rift
        gj = self._grid_nearest_after(tree, d, nearest, gap, keep, new_pos)
        self.pos = np.vstack([self.pos[keep], new_pos])
        self._gj, self._gj_pos = gj, self.pos
        self.plate = np.concatenate([self.plate[keep], new_plate]).astype(np.int32)
        self.cont = np.concatenate([self.cont[keep], new_cont])
        self.birth = np.concatenate([self.birth[keep], np.where(new_cont, np.nan, float(t_new))])
        self.block = np.concatenate([self.block[keep], np.full(n_new, -1, np.int32)])
        for name, (arr, nv) in self.extra.items():
            self.extra[name] = (np.concatenate([arr[keep], np.full(n_new, nv, dtype=arr.dtype)]), nv)
        if "cont_src" in self.extra and new_cont.any():                # origin label of rift-stretched new crust
            j_ = int(keep.sum()) + np.nonzero(new_cont)[0]
            self.extra["cont_src"][0][j_] = CONT_SRC_RIFT; self.extra["cont_since"][0][j_] = t_new
        self.log.append({"t": t_new, "new_ocean_cells": n_new, "consumed_cells": n_consumed,
                         "particles": int(len(self.pos))})

    def merge_slivers(self, min_cells=30, next_id=None):
        """A plate must be one connected piece (RULES.md A2). Small isolated oceanic patches are numerical slivers:
        they go to the plate that surrounds them (continental crust is never relabelled). A larger detached piece is a
        real plate split: it becomes a new plate. Returns a list of (old_id, new_id) splits."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        st = self.grid_state()
        plate = st["plate"].copy()
        if not hasattr(self, "_grid_nb"):
            self._grid_nb = self.grid_tree.query(self.grid, k=7)[1][:, 1:]
        nb = self._grid_nb
        # smooth plate boundaries at the grid scale: an oceanic cell with >= 4 of its 6 neighbours on one other plate
        # joins that plate (one-cell-wide intrusions are numerical, real boundaries are smooth at ~100 km)
        for _ in range(2):
            nbp = plate[nb]
            for k in range(nb.shape[1]):
                q = nbp[:, k]
                cnt = (nbp == q[:, None]).sum(axis=1)
                flip = (~st["cont"]) & (q != plate) & (cnt >= 4)
                plate[flip] = q[flip]
        smoothed = plate.copy()
        plate_before = st["plate"]
        i = np.repeat(np.arange(len(self.grid)), nb.shape[1]); j = nb.ravel()
        same = plate[i] == plate[j]
        g = coo_matrix((np.ones(same.sum()), (i[same], j[same])), shape=(len(self.grid),) * 2)
        ncomp, label = connected_components(g, directed=False)
        size = np.bincount(label, minlength=ncomp)
        cont_in = np.bincount(label, weights=st["cont"].astype(float), minlength=ncomp)
        # small detached pieces (ocean or continental crumbs) join the surrounding plate; continental crust keeps its
        # type, it is only accreted to the neighbour (terrane accretion), never deleted (A10)
        small = size < min_cells
        # detached pieces: every component that is not the largest component of its plate
        comp_plate = np.zeros(ncomp, np.int64); comp_plate[label] = plate
        largest = {}
        for c_ in range(ncomp):
            p_ = comp_plate[c_]
            if p_ not in largest or size[c_] > size[largest[p_]]:
                largest[p_] = c_
        detached = np.array([c_ != largest[comp_plate[c_]] for c_ in range(ncomp)])
        splits = []
        new_plate = plate.copy()
        if next_id is not None:
            for c_ in np.nonzero(detached & ~small)[0]:
                nid = next_id(); splits.append((int(comp_plate[c_]), int(nid)))
                new_plate[label == c_] = nid
        if not small.any() and not splits:
            return splits
        for c in np.nonzero(small)[0]:
            cells = np.nonzero(label == c)[0]
            around = plate[nb[cells].ravel()]
            around = around[around != plate[cells[0]]]
            if around.size:
                new_plate[cells] = np.bincount(around - around.min()).argmax() + around.min()
        # relabel the particles that sit on the changed grid cells (continental particles only follow a split)
        changed = new_plate != plate_before
        near = self.grid_tree.query(self.pos)[1]
        split_cells = np.zeros(len(self.grid), bool)
        for old, nid in splits:
            split_cells |= new_plate == nid
        move = changed[near]
        self.plate[move] = new_plate[near[move]]
        return splits

    def close_pockets(self, max_cells=5, t=np.nan):
        """Ocean pockets of at most max_cells grid cells enclosed by the continental crust of one plate become continental
        crust (suture). They are gaps one or two cells wide left where pieces of a continent were cut apart and joined
        again; an ocean basin needs a ridge and two flanks, so a basin this small is below what the grid can represent,
        and in nature the ocean between colliding margins is consumed or squeezed into the suture (ophiolites), not left
        open. Checked against Earth: enclosed ocean pockets in Müller 2022 continents at the same cell size: 19, 1.4% of
        the crust area (earth_pangea_stats.py; before this rule the planets had 95-130). Returns cells closed."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        st = self.grid_state()
        if not hasattr(self, "_grid_nb"):
            self._grid_nb = self.grid_tree.query(self.grid, k=7)[1][:, 1:]
        nb, oc = self._grid_nb, ~st["cont"]
        idx = np.nonzero(oc)[0]
        pos = -np.ones(len(self.grid), int); pos[idx] = np.arange(len(idx))
        ii = np.repeat(np.arange(len(idx)), nb.shape[1]); jj = pos[nb[idx].ravel()]; ok = jj >= 0
        lab = connected_components(coo_matrix((np.ones(ok.sum()), (ii[ok], jj[ok])), shape=(len(idx),) * 2), directed=False)[1]
        size = np.bincount(lab)
        close = np.zeros(len(self.grid), bool); to_plate = np.zeros(len(self.grid), np.int32)
        for c_ in np.nonzero(size <= max_cells)[0]:
            cells = idx[lab == c_]
            rim = np.setdiff1d(nb[cells].ravel(), cells)
            if st["cont"][rim].all() and len(np.unique(st["plate"][rim])) == 1:
                close[cells] = True; to_plate[cells] = st["plate"][rim[0]]
        if close.any():
            near = self.grid_tree.query(self.pos)[1]
            m = close[near] & ~self.cont
            self.cont[m] = True
            self.birth[m] = np.nan
            if "cont_src" in self.extra:                               # origin label (t: time of the call)
                self.extra["cont_src"][0][m] = CONT_SRC_POCKET; self.extra["cont_since"][0][m] = t
            self.plate[m] = to_plate[near[m]]
            self._gj = None
        return int(close.sum())

    def _grid_nearest_after(self, tree, d, nearest, gap, keep, new_pos):
        """Nearest particle of every grid point after a step's removals and additions, from the step's own search
        (same answer as a fresh tree on the new particles, without building it: saves a tree and a full search)."""
        n_keep = int(keep.sum())
        newidx = np.cumsum(keep) - 1
        j = np.where(keep[nearest], newidx[nearest], -1)
        dist = np.where(keep[nearest], d, np.inf)
        g = np.nonzero(gap)[0]
        j[g] = n_keep + np.arange(len(g)); dist[g] = 0.0                       # new particles sit on these grid points
        lost = np.nonzero(j < 0)[0]                                            # nearest particle was consumed
        if len(lost):
            dk, kk = tree.query(self.grid[lost], k=16)
            ok = keep[np.minimum(kk, len(keep) - 1)] & (kk < len(keep))
            first = ok.argmax(axis=1)
            has = ok.any(axis=1)
            j[lost[has]] = newidx[kk[has, first[has]]]; dist[lost[has]] = dk[has, first[has]]
            rest = lost[~has]
            if len(rest):                                                      # rare: search the kept particles
                dr, jr = cKDTree(self.pos[keep]).query(self.grid[rest])
                j[rest] = jr; dist[rest] = dr
        if len(g):                                                             # a new particle may be nearer
            others = np.nonzero(~gap)[0]
            dn, jn = cKDTree(new_pos).query(self.grid[others], distance_upper_bound=float(dist[others].max()))
            closer = dn < dist[others]
            j[others[closer]] = n_keep + jn[closer]
        return j

    def grid_state(self):
        """State on the check grid: each grid point takes its nearest particle (one plate everywhere)."""
        if getattr(self, "_gj", None) is not None and self._gj_pos is self.pos:
            j = self._gj
        else:
            _, j = cKDTree(self.pos).query(self.grid)
        out = {"plate": self.plate[j], "cont": self.cont[j], "birth": self.birth[j], "block": self.block[j]}
        for name, (arr, _) in self.extra.items():
            out[name] = arr[j]
        return out

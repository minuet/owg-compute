"""Terrain from the plate state (RULES.md A12, A14, A21, A22).

Continents: crustal thickness by tectonic setting, turned into elevation with Earth's measured thickness->elevation
relation (CRUST1.0 medians, results/crust_elevation_curve.json). Isostasy depends on density ratios, not gravity,
so the relation carries over to our planet.
  - craton core / platform: 42 / 37.5 km (Christensen & Mooney 1995: average 41.1 km; CRUST1.0 continental mean 37.6)
  - passive margins: coastal plain -> shelf (~30 km) -> slope -> continent-ocean transition (~15 km)
  - collision orogens: up to +24 km excess (~65 km crust, Tibet 60-80 km) when formed; the excess decays in two stages
    (orogen_excess_now): e-folding 40 Myr while above 8 km (Pelletier 2004: ~50 Myr; Baldwin et al. 2003: stream-power
    decay 1-10 Myr, up to 6x longer with isostasy), then 150 Myr (Baldwin et al. 2003: residual relief for hundreds of
    Myr; Appalachians / Urals)
  - continental arcs behind active trenches: Andean thickening, cap +30 km (~70 km crust; Central Andes 70-74 km,
    Beck et al.); island arcs 15-36 km (Izu-Bonin ~20, Aleutians 35-37)
Oceans: GDH1 (Stein & Stein 1992) + GlobSed sediment loading (Straume et al. 2019); trenches deepened near
convergent boundaries. Sea level: the level that leaves 80% of the surface under water (A20, B3).
"""
import json, os, math
import numpy as np
from fastkd import cKDTree
from planet_checks import boundary_pairs, unit
from ocean_depth_check import gdh1, sediment_thickness, sediment_shallowing
from plate_checks import HERE
from plate_engine import xyz_to_latlon

CURVE = json.loads((HERE / "results/crust_elevation_curve.json").read_text())["thickness_km_vs_median_elevation_m"]
CURVE_T = np.array([c[0] for c in CURVE], float)
CURVE_E = np.array([c[1] for c in CURVE], float)
OROGEN_TAU_MYR = 150.0
def orogen_excess_now(ex, age):
    """Collision-belt excess thickness (km) after age Myr: e-folding 40 Myr while above 8 km, then 150 Myr (see above)."""
    ex = np.asarray(ex, float); age = np.maximum(np.asarray(age, float), 0.0)
    t_fast = np.where(ex > 8.0, 40.0 * np.log(np.maximum(ex, 8.0) / 8.0), 0.0)
    return np.where(age < t_fast, ex * np.exp(-age / 40.0), np.minimum(ex, 8.0) * np.exp(-np.maximum(age - t_fast, 0) / OROGEN_TAU_MYR))


RESID = json.loads((HERE / "results/crust_residual_quantiles.json").read_text())


def continental_residual(th_km, u):
    """Elevation residual (m) for crust of thickness th at quantile u (0..1), from Earth's CRUST1.0 residuals
    (surface minus the thickness-median curve). The residual is mostly regional mantle support / lithosphere
    temperature, so u comes from a smooth spatial field (one region is high or low as a whole)."""
    lv = np.array(RESID["levels_pct"]) / 100.0
    out = np.zeros(len(th_km))
    for c in RESID["classes"]:
        lo, hi = c["thickness_km"]
        sel = (th_km >= lo) & (th_km < hi) if hi < 80 else (th_km >= lo)
        if lo == 15:
            sel = th_km < hi
        out[sel] = np.interp(u[sel], lv, c["quantiles_m"])
    return out


def elevation_from_thickness(th_km):
    return np.interp(th_km, CURVE_T, CURVE_E)


MARGIN = json.loads((HERE / "results/earth_margin_stats.json").read_text())
ISLAND = json.loads((HERE / "results/earth_island_stats.json").read_text())


def margin_profile(d_km, cls):
    """Crustal thickness vs distance inland from the continent-ocean boundary, Earth's measured median and spread
    (IQR/1.35) for class cls ("passive_craton", "passive_noncraton", "active"): CRUST1.0 with the same definitions as
    here (continental crust >= 15 km, distance to the nearest thinner-crust cell, active = < 400 km behind a 0 Ma
    trench; tools/earth_margin_stats.py)."""
    b, rows = MARGIN["bins_km"], MARGIN[cls]
    d = [(a + z) / 2 for a, z, r in zip(b[:-1], b[1:], rows) if r]
    med = np.interp(d_km, d, [r["thickness_median_km"] for r in rows if r])
    sd = np.interp(d_km, d, [(r["thickness_p25_p75"][1] - r["thickness_p25_p75"][0]) / 1.35 for r in rows if r])
    return med, sd


def to_earth_quantiles(v, stats):
    """Quantile mapping: keep the order of v (where it is higher or lower comes from the model) and give it Earth's
    measured distribution (stats: levels_pct + quantiles_m from earth_island_stats.json)."""
    from scipy.stats import rankdata
    if len(v) == 0:
        return v
    r = (rankdata(v) - 0.5) / len(v) * 100.0
    return np.interp(r, ISLAND["levels_pct"], stats["quantiles_m"])


def build_terrain(grid, st, w, t, radius_km, ocean_fraction=0.80, rng=None, mantle=None):
    lat, lon = xyz_to_latlon(grid)
    cont = st["cont"]
    n = len(grid)
    chord_to_km = lambda c: 2 * np.arcsin(np.clip(c / 2, 0, 1)) * radius_km

    # boundary kinematics at time t
    plate = st["plate"]
    W = np.array([w.get(int(p), np.zeros(3)) for p in plate])
    i, j = boundary_pairs(grid, plate)
    mid = unit(grid[i] + grid[j]); nrm = unit(grid[j] - grid[i])
    opening = np.einsum("ij,ij->i", (np.cross(W[j], mid) - np.cross(W[i], mid)) * radius_km / 10.0, nrm)
    conv = opening < -0.5
    age = st["birth"] - t
    ai, aj = np.nan_to_num(age[i], nan=1e6), np.nan_to_num(age[j], nan=1e6)
    i_over = cont[i] | (~cont[j] & (ai < aj))
    over_pts = np.where(i_over, i, j)[conv]
    sub_pts = np.where(i_over, j, i)[conv]

    # distance of every point to the nearest ocean point (for continents) and to the convergent boundary
    ocean_tree = cKDTree(grid[~cont]) if (~cont).any() else None
    d_coast = np.zeros(n)
    if ocean_tree is not None:
        d_coast[cont] = chord_to_km(ocean_tree.query(grid[cont])[0])
    d_trench = np.full(n, np.inf)
    if len(over_pts):
        d_trench = chord_to_km(cKDTree(mid[conv]).query(grid)[0])
    over_side = np.zeros(n, bool); over_side[over_pts] = True
    over_side = cKDTree(grid[over_pts]).query(grid)[0] < cKDTree(grid[sub_pts]).query(grid)[0] if len(over_pts) else over_side

    # smooth regional field (unit spread) for where the crust is thicker or thinner than its class median
    r_ = rng if rng is not None else np.random.default_rng(0)
    smooth = np.zeros(n)
    for _ in range(16):
        d = r_.normal(size=3); d /= np.linalg.norm(d)
        smooth += np.sin(r_.uniform(10, 25) * (grid @ d) + r_.uniform(0, 2 * math.pi))
    smooth /= smooth.std()
    core = st.get("core", np.zeros(n)) > 0.5
    # normal crust by province and distance from the ocean, as measured on Earth (CRUST1.0 passive margins, split into
    # cratons - Müller 2022 craton polygons - and the rest; earth_margin_stats.py): median of the class at that distance,
    # plus its spread (IQR/1.35) times a smooth field, so whole regions are thick or thin together. Earth's crust
    # thickens inland over ~1500 km (cratons 36 km at 300-400 km, 42 km beyond 1500 km). Core cells = cratons.
    # Collision belts and arcs are added on top. 2026-10-31: replaces fixed 42 / 37.5 km (Christensen & Mooney 1995)
    # capped by a hand-made margin profile whose widths had been set to fix the sea level (trial and error).
    active_margin = cont & (d_trench < 400) & over_side
    if os.environ.get("DEBUG_MARGIN"):
        for D_ in (150, 300):
            m_ = cont & (d_coast < D_)
            print(f"margin: continental cells within {D_} km of ocean crust that are active: {active_margin[m_].mean()*100:.1f}% "
                  f"(Earth 19.1% / 20.6%)", flush=True)
    th = np.zeros(n)
    for cls, m_ in (("passive_craton", cont & core & ~active_margin), ("passive_noncraton", cont & ~core & ~active_margin),
                    ("active", cont & active_margin)):
        med, sd = margin_profile(d_coast[m_], cls)
        th[m_] = med + sd * np.clip(smooth[m_], -2.5, 2.5)
    if "rift_thin" in st:
        th -= st["rift_thin"]
    if "orogen_time" in st:
        ot, ox = st["orogen_time"], st["orogen_excess"]
        ok = np.isfinite(ot)
        age_o = ot[ok] - t
        th[ok] += orogen_excess_now(ox[ok], age_o)
    arc = cont & (st["arc_excess"] > 1.0) if "arc_excess" in st else np.zeros(n, bool)
    # (2026-11-01: the subduction-margin thickness change (arc_excess: Andean thickening / back-arc thinning) is no longer
    # added. Earth test (earth_arc_check.py, Müller 2022 200 Ma -> now): the rule changed 31% of Earth's continental
    # crust by a median -5.1 / +5.9 km, but CRUST1.0 there is no thinner / thicker than elsewhere at the same distance
    # from the ocean (+1.0 / +1.1 km vs +1.0 km untouched, correlation 0.07); Earth's measured margin profiles and
    # residuals above already hold the arcs. arc_excess stays only as a record of where margins were active.)
    # active continental rifts at time t: continent-continent boundaries that are opening (East African type).
    # Rift floor (graben) ~1 km down within ~40 km of the axis, flanks uplifted ~1-2 km over ~40-120 km
    # (flexural rift-flank uplift, Weissel & Karner 1989); crust under the axis thinned to ~32 km.
    rift_axis = None
    cc_open = (opening > 0.2) & cont[i] & cont[j]
    rift_dist = np.full(n, np.inf)
    # graben half-width, at least about one cell. Real grabens are 40-75 km across (Ebinger 1989), i.e. half-width
    # 20-38 km, below the grid (60000 cells on R = 6000 km: 0.55 x 87 km = 48 km), so the cell term always wins and the
    # modelled graben is ~1.3-2.4x too wide - a resolution limit (audit 2026-11-11), not a choice of the 40
    half_w = max(40.0, 0.55 * math.sqrt(4 * math.pi / n) * radius_km)
    if cc_open.any():
        rift_axis = mid[cc_open]
        rift_dist = chord_to_km(cKDTree(rift_axis).query(grid)[0])
        axis_zone = cont & (rift_dist < half_w)
        th = np.where(axis_zone, np.minimum(th, 32.0), th)
    elev = np.where(cont, elevation_from_thickness(th), np.nan)
    if rift_axis is not None:
        graben = cont & (rift_dist < half_w)
        flank = cont & (rift_dist >= half_w) & (rift_dist < half_w + 120)
        elev = np.where(graben, elev - 1000.0 * (1 - rift_dist / half_w) * 0.5 - 500.0, elev)
        elev = np.where(flank, elev + 1500.0 * np.exp(-((rift_dist - half_w - 40.0) / 40.0) ** 2), elev)

    # oceans
    oc = ~cont
    a = np.clip(np.nan_to_num(age, nan=150.0), 0, 250)
    depth = gdh1(a) - sediment_shallowing(np.minimum(sediment_thickness(a, lat), 1300))
    trench = oc & ~over_side & (d_trench < 80)
    # trenches lie 3-4 km below the surrounding ocean floor and are 50-100 km wide (Kearey, Klepeis & Vine 2009)
    depth = depth + np.where(trench, 3500 * (1 - d_trench / 80), 0)
    # arc ridge behind the trench: volcanic front 166 +- 60 km from the trench (Gill 1981; Dickinson 1973: 100-300 km)
    island_arc = oc & over_side & (d_trench > 100) & (d_trench < 240)
    # island-arc crust ~20-35 km (Izu-Bonin ~20-22, Aleutians ~30-35). Arc maturity varies along strike in whole
    # segments (Japan, Philippines, Java emerge as large islands; Izu-Bonin, Tonga, South Sandwich stay mostly
    # submarine), so the crust thickness is a smooth along-arc field (segments ~300-3000 km), not cell-by-cell noise
    # (cell noise made every trench a string of evenly spaced dots; fixed 2026-10-26). Same field on any grid.
    ra = rng if rng is not None else np.random.default_rng(2)
    seg = np.zeros(n); seg_s = np.zeros(n)
    for _ in range(16):
        d = ra.normal(size=3); d /= np.linalg.norm(d)
        seg += np.sin(ra.uniform(12, 60) * (grid @ d) + ra.uniform(0, 2 * math.pi))
    for _ in range(16):
        d = ra.normal(size=3); d /= np.linalg.norm(d)
        seg_s += np.sin(ra.uniform(60, 250) * (grid @ d) + ra.uniform(0, 2 * math.pi))
    seg /= math.sqrt(16 / 2); seg_s /= math.sqrt(16 / 2)
    mature = 1 / (1 + np.exp(-(seg - 1.2) * 5.0))          # ~12% of arc length mature enough to emerge
    arc_elev = elevation_from_thickness(np.clip(19.0 + 12.0 * mature + 1.5 * seg_s, 15.0, 36.0))
    # only the volcanic front (~120-220 km behind the trench, as in lithology and the volcano list) can emerge
    arc_elev = np.where((d_trench > 120) & (d_trench < 220), arc_elev, np.minimum(arc_elev, elevation_from_thickness(np.full(n, 21.0))))
    elev = np.where(oc, -depth, elev)
    elev = np.where(island_arc, np.maximum(elev, arc_elev), elev)

    if mantle is not None:
        rng2 = rng if rng is not None else np.random.default_rng(0)
        # oceanic plateaus (plume heads): tops ~2-2.5 km below sea level (Ontong Java-like), A17
        if "plateau" in st:
            pf = np.zeros(n)
            for f in np.geomspace(20, 160, 40):
                d = rng2.normal(size=3); d /= np.linalg.norm(d)
                pf += (f / 20) ** -0.9 * np.sin(f * (grid @ d) + rng2.uniform(0, 2 * math.pi))
            pf /= pf.std()
            pl = oc & (st["plateau"] > 0.5) & (pf > -0.6)          # ragged outline, not a disc
            elev = np.where(pl, np.maximum(elev, -2300 + 450 * pf), elev)
        # dynamic topography (Hoggard et al. 2016): ~±500 m at ~10^4 km (degree-2: upwelling under the old
        # supercontinent and its antipode, downwelling around the girdle) plus ~±1 km peaks at ~10^3 km
        c = mantle["upwelling_centre"]
        mu = grid @ c
        dyn = 500.0 * (3 * mu ** 2 - 1) / 2.0
        # regional undulation as a fractal field (many waves, amplitude ~ wavelength, i.e. power ~ k^-2: Earth's
        # topography spectrum, Turcotte "Fractals and Chaos in Geology and Geophysics"), wavelengths ~2700 km down to
        # ~2 grid cells (2026-10-31: was cut at ~300 km, and a separate unsourced "near-shore relief" of 160 m supplied
        # the shorter scales; the measured residual distribution below already holds Earth's variance at all scales
        # CRUST1.0 resolves, so one field with Earth's spectrum replaces both)
        field = np.zeros(n)
        for f in np.geomspace(14, 220, 140):          # 220: two cells of the 60000-point grid; fixed, so the fine
                                                      # replay grid gets the same field
            d = rng2.normal(size=3); d /= np.linalg.norm(d)
            field += (f / 14) ** -1.0 * np.sin(f * rng2.uniform(0.9, 1.1) * (grid @ d) + rng2.uniform(0, 2 * math.pi))
        field *= 330.0 / field.std()
        dyn += np.clip(field, -1000, 1000)
        # on continents: the thickness-median curve already holds the average mantle support; what it leaves out is
        # Earth's measured residual (CRUST1.0: sigma ~450 m at 35-45 km, ~700-1000 m for thicker crust, skewed upward).
        # A smooth field (the degree-2 mantle pattern plus regional undulation) sets each region's quantile in that
        # residual distribution. Validated on Earth's own crust (tools/results: method reproduces Earth's hypsometry).
        from scipy.stats import norm
        z = dyn / max(dyn.std(), 1.0)
        # no extra inland rise: on Earth the residual has no trend with distance from the ocean (CRUST1.0 passive
        # margins: -40 m at 0-150 km, -14..+20 m from 150 to 3000 km; earth_margin_stats.py). Interiors stand higher
        # only because the crust is thicker (margin_profile). 2026-10-31: removed an unsourced shift 1.5*tanh(...)
        u = norm.cdf((z - z[cont].mean()) / max(z[cont].std(), 1e-6)) if cont.any() else np.full(n, 0.5)
        dyn = np.where(cont, continental_residual(th, np.clip(u, 0.001, 0.999)), dyn)
        # hotspot swells (Sleep 1990: 500-1200 m high, ~1000-1500 km wide), and the swell left behind along the track
        hs = mantle["hotspots"]; amp = mantle["hot_strength"]
        dkm = 2 * np.arcsin(np.clip(np.linalg.norm(grid[:, None, :] - hs[None, :, :], axis=2) / 2, 0, 1)) * radius_km
        swell = (amp[None, :] * np.exp(-(dkm / 450.0) ** 2)).max(axis=1)
        if "hs_time" in st:
            age_h = st["hs_time"] - t
            trail = np.where(np.isfinite(age_h), st["hs_amp"] * np.exp(-np.nan_to_num(age_h, nan=1e9) / 30.0), 0.0)
            swell = np.maximum(swell, trail)
            # volcanic chain: only the youngest volcanoes (~5 Myr, Hawaii-like) stand above the sea; older ones subside
            # and erode into seamounts and guyots ~1-2 km below sea level
            vol = oc & np.isfinite(age_h)
            ah = np.nan_to_num(age_h, nan=1e9)
            # separate volcanoes along the track (one every ~1.5 Myr of plate motion, Hawaii-like), not a continuous
            # ridge; heights are cell means (a 20 km cell around a 4 km peak averages ~1 km)
            beads = np.cos(2 * math.pi * ah / 1.5) ** 6
            peak = -depth + (depth + 1200) * np.exp(-ah / 4.0) * beads - 600 * (1 - beads)
            elev = np.where(vol, np.maximum(elev, np.minimum(peak, 1500)), elev)
        elev = elev + dyn + swell
        # oceanic crust within 500 km of an active hotspot: Earth's measured distribution of cell-mean elevation there
        # (ETOPO5 at this cell size, 42 oceanic hotspots: 0.47% of cells above sea level; earth_island_stats.py). The
        # volcanoes, beads and swell above only decide which cells are the high ones. 2026-10-31: their own amplitudes
        # (unsourced) made 6% of these cells land, 13x Earth
        d_hot = dkm.min(axis=1)
        hz = oc & (d_hot < 500.0) & ~island_arc
        elev[hz] = to_earth_quantiles(elev[hz], ISLAND["hotspot_500km"])
    # island arcs on oceanic crust 100-240 km behind the trench: Earth's measured distribution there (oceanic overriding
    # plates of the Müller 2022 0 Ma trenches: 1.7% of cells above sea level; earth_island_stats.py); the along-arc
    # maturity field above only decides which segments are high. 2026-10-31: replaces an unsourced ~12% emergent share
    if island_arc.any():
        elev[island_arc] = to_earth_quantiles(elev[island_arc], ISLAND["island_arc_belt"])

    # sediment-built continental shelves on passive margins: rivers fill the thinned, subsiding margin up to near wave
    # base, so the shelf runs from the shore (0 m) to the shelf break (~ -130 m; Shepard: mean break ~130-140 m, mean
    # width ~80 km). Width varies along the coast (30-250 km). Median 74.4 km: the lognormal (sigma 0.6) clipped to
    # 30-250 km then has mean 88 km, Harris et al. 2014's passive-margin mean (PARAM_AUDIT 2026-11-11; was median 80, mean 94). Active margins have narrow shelves (below grid scale).
    rr = rng if rng is not None else np.random.default_rng(1)
    wfield = np.zeros(n)
    for _ in range(12):
        d = rr.normal(size=3); d /= np.linalg.norm(d)
        wfield += np.sin(rr.uniform(8, 20) * (grid @ d) + rr.uniform(0, 2 * math.pi))
    shelf_w = np.exp(math.log(74.4) + 0.6 * wfield / max(wfield.std(), 1e-6))     # lognormal, median 74.4 km
    shelf_w = np.clip(shelf_w, 30.0, 250.0)
    passive = cont & ~active_margin
    for _ in range(2):
        sl = float(np.quantile(elev, ocean_fraction))
        land = elev > sl
        if not land.any():
            break
        d_shore = chord_to_km(cKDTree(grid[land]).query(grid)[0])
        sea_cont = passive & ~land & (elev > sl - 2500.0) & (d_shore <= shelf_w)
        shelf_depth = sl - 130.0 * np.clip(d_shore / shelf_w, 0, 1)
        elev = np.where(sea_cont, np.maximum(elev, shelf_depth), elev)

    # (2026-10-31: the separate "near-shore relief" of 160 m is removed: no source, and it added variance on top of
    # Earth's measured residual distribution; the short scales are now in the residual field itself)
    # sea level from the water budget: ocean_fraction of the surface under water (grid is area-uniform)
    sea_level = float(np.quantile(elev, ocean_fraction))
    # closed depressions: low ground cut off from the sea cannot be flooded by it. Over geological time such basins
    # fill with water and sediment up to their outlet (spill point) - the standard depression filling of hydrology
    # (priority flood, Barnes et al. 2014). Low ground connected to the open ocean stays sea (shelves, epicontinental
    # seas: Hudson Bay, Baltic). Below-sea pieces of >= 0.2% of the surface count as open sea.
    # 2026-10-30: replaces "fill every low > 100 km from oceanic crust to 30-200 m" (a trial rule): it also raised the
    # flooded shelves and margins, so only ~14% of continental crust stayed under water (Earth 29%; before it ~31%), and
    # the 80% ocean then raised the sea 100-240 m and drowned coastal lowlands.
    import heapq
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nbk = cKDTree(grid).query(grid, k=7)[1][:, 1:]
    for _ in range(2):
        low = elev < sea_level
        idx = np.nonzero(low)[0]
        pos = -np.ones(n, int); pos[idx] = np.arange(len(idx))
        ii = np.repeat(np.arange(len(idx)), 6); jj = pos[nbk[idx].ravel()]
        ok_ = jj >= 0
        _, lab_l = connected_components(coo_matrix((np.ones(ok_.sum()), (ii[ok_], jj[ok_])), shape=(len(idx), len(idx))), directed=False)
        open_sea = np.zeros(n, bool); open_sea[idx[np.bincount(lab_l)[lab_l] >= 0.002 * n]] = True
        filled = np.where(open_sea, elev, np.inf)
        heap = [(float(elev[k]), int(k)) for k in np.nonzero(open_sea)[0]]
        heapq.heapify(heap)
        done = open_sea.copy()
        while heap:
            h, k = heapq.heappop(heap)
            for m in nbk[k]:
                if not done[m]:
                    done[m] = True
                    filled[m] = max(float(elev[m]), h)
                    heapq.heappush(heap, (filled[m], int(m)))
        closed = low & ~open_sea
        elev = np.where(closed, np.maximum(filled, sea_level + 1.0), elev)
        sea_level = float(np.quantile(elev, ocean_fraction))
    elev = elev - sea_level
    return {"elev": elev, "thickness": np.where(cont, th, np.nan), "sea_level_shift_m": sea_level,
            "arc": arc | island_arc, "trench": trench, "rift_dist_km": rift_dist, "rift_half_width_km": half_w,
            "d_trench_km": d_trench, "over_side": over_side,
            "rift_volcanoes": (rift_axis[::max(1, len(rift_axis) // 20)] if rift_axis is not None else np.zeros((0, 3)))}


def hypsometry(elev, lat=None):
    land, ocean = elev > 0, elev <= 0
    f = lambda m, b: float((m & b).sum() / max(b.sum(), 1))
    return {"land_frac_of_surface": float(land.mean()), "land_mean_m": float(elev[land].mean()),
            "land_median_m": float(np.median(elev[land])), "land_above_1000m_frac": f(elev > 1000, land),
            "land_above_2000m_frac": f(elev > 2000, land), "land_above_3000m_frac": f(elev > 3000, land),
            "ocean_mean_depth_m": float(-elev[ocean].mean()), "shelf_0_200m_frac_of_ocean": f(elev > -200, ocean),
            "ocean_shallower_1000m_frac": f(elev > -1000, ocean)}


def update_paleo_records(eng, dt, radius_km, arc_name="arc_excess", flooded=None):
    """Paleo-environment history of continental crust (for lithology and special landforms later, LANDFORMS.md):
    - carb_time: Myr under shallow sea at paleolatitude < 30 deg: a passive-margin shelf (within ~300 km of the ocean,
      not an active margin) or continent flooded by high long-term sea level (epicontinental sea) -> limestone, chalk
    - arid_time: Myr in the subtropical dry belt (15-35 deg) far inland (> 500 km from the sea) -> evaporites, red beds
    - humid_time: Myr in wet lowland belts (equatorial < 12 deg or mid-latitude 40-60 deg) within ~800 km of the sea
      -> coal swamps
    Paleolatitude is the latitude in the mantle frame (stated assumption: no true polar wander)."""
    c = eng.cont
    if not c.any() or (~c).all():
        return
    lat = np.degrees(np.arcsin(np.clip(eng.pos[:, 2], -1, 1)))
    d_sea = 2 * np.arcsin(np.clip(cKDTree(eng.pos[~c]).query(eng.pos[c])[0] / 2, 0, 1)) * radius_km
    al = np.abs(lat[c])
    active = eng.extra[arc_name][0][c] > 1.0 if arc_name in eng.extra else np.zeros(c.sum(), bool)
    shelf_trop = (d_sea < 300) & (al < 30) & ~active
    if flooded is not None:
        shelf_trop = (shelf_trop | flooded[c]) & (al < 30)
    arid = (al >= 15) & (al <= 35) & (d_sea > 500)
    humid = ((al < 12) | ((al >= 40) & (al <= 60))) & (d_sea < 800)
    for name, sel in (("carb_time", shelf_trop), ("arid_time", arid), ("humid_time", humid)):
        if name in eng.extra:
            arr = eng.extra[name][0]
            idx = np.nonzero(c)[0][sel]
            arr[idx] += dt


def particle_elev_proxy(eng, t, radius_km, mantle_centre=None, hotspot_amp=None):
    """Approximate surface height (m, Earth datum) of every tracer particle at time t, cheap enough to call during the
    run (no neighbours): crust thickness from its records -> Earth thickness-elevation curve; ocean from age (GDH1);
    plus degree-2 mantle support and hotspot swells. Used for the vertical-motion history (L2) and for marine flooding
    of continents (carbonates, chalk; L1/L6). Margins and rift flanks are not resolved here."""
    ex = eng.extra
    c = eng.cont
    th = np.where(ex["core"][0] > 0.5, 42.0, 37.5) if "core" in ex else np.full(len(c), 40.0)
    if "rift_thin" in ex:
        th = th - ex["rift_thin"][0]
    if "orogen_time" in ex:
        ot, ox = ex["orogen_time"][0], ex["orogen_excess"][0]
        ok = np.isfinite(ot)
        age_o = np.where(ok, ot - t, 0.0)
        th = th + np.where(ok & (age_o >= 0), orogen_excess_now(ox, age_o), 0.0)
    # (arc_excess not added: failed the Earth test, see build_terrain)
    e = np.where(c, elevation_from_thickness(th), -(gdh1(np.clip(np.nan_to_num(eng.birth - t, nan=150.0), 0, 250))))
    if mantle_centre is not None:
        mu = eng.pos @ mantle_centre
        e = e + 500.0 * (3 * mu ** 2 - 1) / 2.0 * np.where(c, 0.5, 1.0)
    if "hs_time" in ex:
        age_h = ex["hs_time"][0] - t
        e = e + np.where(np.isfinite(age_h) & (age_h >= 0), ex["hs_amp"][0] * np.exp(-np.nan_to_num(age_h, nan=1e9) / 30.0), 0.0)
    return e


def eustatic_sea_level(eng, t, ref_mean_depth=None):
    """Long-term sea level from mid-ocean ridge volume (Pitman 1978): younger ocean floor is shallower and pushes water
    onto the continents. Change relative to the reference (present) mean ocean depth, with the isostatic factor for
    water loading (rho_m - rho_w)/rho_m ~ 0.69. Returns (sea level change in m, mean ocean depth in m)."""
    oc = ~eng.cont
    d = gdh1(np.clip(np.nan_to_num(eng.birth[oc] - t, nan=150.0), 0, 250)).mean()
    if ref_mean_depth is None:
        return 0.0, float(d)
    return float((ref_mean_depth - d) * (3300.0 - 1030.0) / 3300.0), float(d)


def update_basins(eng, dt, ep, flooded, radius_km):
    """Sediment thickness accumulated on continental crust (m), with typical accumulation rates (tens to hundreds of m
    per Myr): epicontinental sea ~15, passive-margin shelf ~30, active rift (rift stage, < 30 Myr since rifting) ~100,
    foreland basin in front of a growing mountain belt ~150; high ground (> 1000 m) loses ~20 m/Myr to erosion."""
    ex = eng.extra
    if "sed_thick" not in ex:
        return
    c = eng.cont
    rate = np.zeros(len(c))
    rate += 15.0 * flooded
    sed = ex["sed_thick"][0]
    if "foreland_time" in ex:
        ft = ex["foreland_time"][0]
        rate += 150.0 * np.isfinite(ft) * (np.nan_to_num(ft, nan=-1e9) >= eng._t_now - 10)
    if "rift_time" in ex:
        rt = ex["rift_time"][0]
        rate += 100.0 * np.isfinite(rt) * (np.nan_to_num(rt, nan=-1e9) <= eng._t_now + 30) * (np.nan_to_num(rt, nan=-1e9) >= eng._t_now)
    oc_tree = cKDTree(eng.pos[~c]) if (~c).any() else None
    if oc_tree is not None and c.any():
        d_sea = 2 * np.arcsin(np.clip(oc_tree.query(eng.pos[c])[0] / 2, 0, 1)) * radius_km
        shelf = np.zeros(len(c), bool); shelf[np.nonzero(c)[0][d_sea < 250]] = True
        rate += 30.0 * shelf
    rate -= 20.0 * (ep > 1000.0)
    sed[c] = np.maximum(sed[c] + rate[c] * dt, 0.0)


def lithology(st, elev, d_trench=None):
    """Surface rock type of each continental cell from its recorded history (classes, in order of priority):
    0 flood basalt, 1 arc volcanic, 2 crystalline (plutonic/metamorphic: exhumed orogen cores and shields without cover),
    3 carbonate, 4 evaporite / red beds, 5 coal-bearing clastics, 6 other clastic sediments. -1 = ocean.
    Thresholds are first estimates, to be checked against GLiM shares (Hartmann & Moosdorf 2012)."""
    c = st["cont"]
    n = len(c)
    out = np.full(n, -1, np.int8)
    sed = st.get("sed_thick", np.zeros(n))
    covered = sed > 300.0
    oro = np.nan_to_num(st.get("orogen_excess", np.zeros(n))) > 6.0
    shield = (st.get("core", np.zeros(n)) > 0.5) & ~covered
    lith = np.full(n, 6, np.int8)
    lith = np.where(covered & (st.get("humid_time", np.zeros(n)) >= 40), 5, lith)
    lith = np.where(covered & (st.get("arid_time", np.zeros(n)) >= 30), 4, lith)
    lith = np.where(covered & (st.get("carb_time", np.zeros(n)) >= 10), 3, lith)
    lith = np.where(shield | (oro & ~covered) | (elev > 2500), 2, lith)
    arc_belt = st.get("arc_excess", np.zeros(n)) > 3.0
    # the volcanic front is a narrow band ~120-220 km from the trench; the rest of an Andean belt exposes batholiths
    # (plutonic), metamorphic rocks and sediments
    front = arc_belt if d_trench is None else arc_belt & (d_trench > 120) & (d_trench < 220)
    lith = np.where(arc_belt & ~front & ~covered, 2, lith)
    lith = np.where(front, 1, lith)
    lith = np.where(st.get("plateau", np.zeros(n)) > 0.5, 0, lith)
    out[c] = lith[c]
    return out


LITHO_NAMES = ["flood basalt", "arc volcanic", "crystalline (plutonic/metamorphic)", "carbonate", "evaporite/red beds",
               "coal-bearing clastics", "other clastics"]


def volcano_inventory(grid, st, terr, hotspots, hot_strength, events, radius_km, spacing_km=100.0):
    """List of volcanic centres (RULES.md L4): arc volcanoes along the volcanic front (one per ~100 km), active hotspots
    and their dated chains, rift volcanoes along active continental rifts, and large igneous provinces (from events)."""
    from plate_engine import xyz_to_latlon
    out = []
    lat, lon = xyz_to_latlon(grid)
    d_tr = terr["d_trench_km"]; over = terr["over_side"]
    front = over & (d_tr > 120) & (d_tr < 220) & (terr["elev"] > -3000)
    idx = np.nonzero(front)[0]
    if len(idx):
        keep = []
        tree = None
        for k in idx[np.argsort(d_tr[idx])]:
            if tree is None or all(np.linalg.norm(grid[k] - grid[j]) * radius_km > spacing_km for j in keep[-200:]):
                keep.append(k)
        for k in keep:
            out.append({"type": "arc" if st["cont"][k] else "island arc", "lat": float(lat[k]), "lon": float(lon[k]),
                        "active": True, "age_myr": 0.0})
    hl = np.degrees(np.arcsin(np.clip(hotspots[:, 2], -1, 1))); ho = np.degrees(np.arctan2(hotspots[:, 1], hotspots[:, 0]))
    for h in range(len(hotspots)):
        out.append({"type": "hotspot", "lat": float(hl[h]), "lon": float(ho[h]), "active": True, "age_myr": 0.0,
                    "swell_m": float(hot_strength[h])})
    if "hs_time" in st:
        ht = st["hs_time"]
        chain = np.nonzero(np.isfinite(ht) & (ht > 2))[0]
        for k in chain[::3]:
            out.append({"type": "hotspot chain (extinct)", "lat": float(lat[k]), "lon": float(lon[k]), "active": False,
                        "age_myr": float(ht[k])})
    for p in terr.get("rift_volcanoes", np.zeros((0, 3))):
        out.append({"type": "rift", "lat": float(np.degrees(np.arcsin(np.clip(p[2], -1, 1)))),
                    "lon": float(np.degrees(np.arctan2(p[1], p[0]))), "active": True, "age_myr": 0.0})
    for e in events:
        if e["type"] == "large igneous province":
            out.append({"type": "large igneous province", "detail": e["detail"], "active": False, "age_myr": float(e["t"])})
    return out

"""Hydrology on an equirectangular grid (PIPELINE step 8, LANDFORM_INPUTS L8).

Steps (all standard; the professional parts use pysheds):
1. Runoff from annual climate: potential evaporation from Holdridge (1967) biotemperature, PET = 58.93 * Tbio mm/yr;
   actual evaporation from the Budyko-type Fu / Zhang equation (Zhang et al. 2004, w = 2.6); runoff = P - AET.
2. Depressions: priority-flood filling (Barnes et al. 2014, pysheds), with the sea as the outlet everywhere.
3. Flow: flats resolved (Barnes et al. 2015, pysheds) and D8 directions (pysheds) on the filled surface.
4. Which depressions hold a closed (endorheic) lake: water balance of the terminal lake (Mason et al. 1994): water
   reaching a depression is kept if open-water evaporation minus rain over the depression's area at its spill level
   can remove it; otherwise the lake overflows and the excess continues downstream. Processed upstream to downstream,
   so a closed basin's water never reaches the next one.
Rows are latitudes from north to south; column 0 must be a meridian without land where possible (the seam).
"""
import math
import numpy as np
from scipy import ndimage
from affine import Affine
from pysheds.grid import Grid
from pysheds.sview import Raster, ViewFinder

OCEAN_Z = -10000.0
SED_PER_WATER = 3e-4                                      # deposited sediment volume per volume of river water
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)                     # pysheds default: N NE E SE S SW W NW
DR = {64: -1, 128: -1, 1: 0, 2: 1, 4: 1, 8: 1, 16: 0, 32: -1}
DC = {64: 0, 128: 1, 1: 1, 2: 1, 4: 0, 8: -1, 16: -1, 32: -1}


def runoff_mm(P, T, A, w=2.6):
    """Annual runoff (mm/yr) and potential evaporation from annual mean T, seasonal half-range A (deg C) and P (mm/yr)."""
    months = np.cos(2 * np.pi * (np.arange(12) + 0.5) / 12)
    tb = np.clip(T[..., None] + A[..., None] * months, 0, 30).mean(axis=-1)
    pet = 58.93 * tb
    phi = pet / np.maximum(P, 1.0)
    aet = P * (1 + phi - (1 + phi ** w) ** (1 / w))
    return np.maximum(P - aet, 0.0), pet


def hydrology(elev, ocean, P, T, A, radius_km, pad_frac=0.125, sed_myr=0.3):
    """elev (m), ocean (bool), climate on the same (H, W) grid. Returns flow fields and the endorheic mask.
    The grid wraps in longitude: a strip from the other side is added on both sides, so that rivers and basins across
    the seam are handled like everywhere else, and is cut off again at the end."""
    H, W = elev.shape
    pad = max(1, int(W * pad_frac))
    wrap = lambda a: np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)
    r = _hydrology(wrap(elev), wrap(ocean), wrap(P), wrap(T), wrap(A), radius_km, W, sed_myr)
    out = {k: (v[:, pad:pad + W] if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in r.items()}
    out["n_closed"] = int(len(np.unique(out["lake_id"][out["lakes"]])))
    return out


def _topological_fronts(down, has, oc, n):
    """Cells in waves from the sources downstream (Kahn's algorithm, vectorised): every cell comes after all cells
    that drain into it."""
    indeg = np.bincount(down[has], minlength=n)
    front = np.nonzero((indeg == 0) & ~oc)[0]
    order = []
    while len(front):
        order.append(front)
        f = front[has[front]]
        dn = down[f]
        np.subtract.at(indeg, dn, 1)
        nxt = np.unique(dn[indeg[dn] == 0])
        front = nxt[~oc[nxt]]
    return order


def _hydrology(elev, ocean, P, T, A, radius_km, W_true, sed_myr):
    H, W = elev.shape
    lat = 90 - (np.arange(H) + 0.5) * 180 / H
    area = (np.cos(np.radians(lat))[:, None] * np.ones((1, W))) * (2 * math.pi * radius_km / W_true) * (math.pi * radius_km / H)
    R, pet = runoff_mm(P, T, A)
    vf = ViewFinder(affine=Affine(360 / W_true, 0, -180, 0, -180 / H, 90), shape=(H, W), nodata=np.nan)
    grid = Grid(viewfinder=vf)
    dem = Raster(np.where(ocean, OCEAN_Z, elev).astype(np.float64), viewfinder=vf)
    filled = np.asarray(grid.fill_depressions(dem))
    depth = np.where(ocean, 0.0, filled - elev)
    # the sea gets a slope away from the coast, so it is not one huge flat for the flat-resolution step
    off = ndimage.distance_transform_edt(ocean)
    flat_ok = Raster(np.where(ocean, OCEAN_Z - off, filled), viewfinder=vf)
    inflated = grid.resolve_flats(flat_ok)
    fdir = np.asarray(grid.flowdir(inflated, dirmap=DIRMAP))

    # downstream index of every cell (-1: ocean or no outlet)
    idx = np.arange(H * W).reshape(H, W)
    down = np.full(H * W, -1, np.int64)
    for code in DIRMAP:
        rr, cc = np.nonzero(fdir == code)
        r2, c2 = rr + DR[code], cc + DC[code]
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        down[idx[rr[ok], cc[ok]]] = idx[r2[ok], c2[ok]]
    oc = ocean.ravel()
    down[oc] = -1
    down[(down >= 0) & oc[np.maximum(down, 0)]] = -2           # flows into the sea

    n = H * W
    has = down >= 0
    order = _topological_fronts(down, has, oc, n)
    a_ = area.ravel()

    # depressions: connected flooded areas at spill level. A depression is filled by the sediment its inflowing rivers
    # bring: global rivers carry ~19 Gt/yr (Milliman & Farnsworth 2011) in ~38,000 km3/yr of runoff, i.e. a deposited
    # volume of ~3e-4 of the water volume (bulk density ~1.6 t/m3). Over sed_myr million years, depressions smaller
    # than that are infilled and no longer block drainage; large (tectonic) or dry (little water, little sediment)
    # basins stay. Water entering is counted before any evaporation (upper bound).
    lab, nd = ndimage.label(depth > 0.5, structure=np.ones((3, 3)))
    lab = lab.ravel()
    vol = np.bincount(lab, weights=depth.ravel() * a_, minlength=nd + 1)          # m * km2
    catch = a_.copy()
    qin = R.ravel() * a_ * 1e-6                                                   # km3 / yr
    for f in order:
        f = f[has[f]]
        np.add.at(catch, down[f], catch[f])
        np.add.at(qin, down[f], qin[f])
    exit0 = has & (lab[np.maximum(down, 0)] != lab) & (lab > 0)
    exit0 |= (lab > 0) & ~has
    q_lab = np.zeros(nd + 1)
    np.maximum.at(q_lab, lab[exit0], qin[exit0])
    keep_dep = vol * 1e-3 > SED_PER_WATER * q_lab * sed_myr * 1e6                  # km3
    keep_dep[0] = False
    lab = np.where(keep_dep[lab], lab, 0)
    n_sed_filled = int((~keep_dep[1:]).sum())

    # water: runoff outside lakes, rain minus open-water evaporation on the lake area; processed upstream -> downstream
    src = np.where(lab > 0, (P - pet).ravel(), R.ravel()) * a_ * 1e-3                 # m * km2 -> 1e6 m3 / yr units
    src[oc] = 0.0
    water = src.copy()
    closed = np.zeros(nd + 1, bool)
    exit_cell = has & (lab[np.maximum(down, 0)] != lab) & (lab > 0)   # the lake's spill cell
    exit_cell |= (lab > 0) & ~has                                      # a lake with no outlet at all
    for front in order:
        ex = front[exit_cell[front]]
        if len(ex):
            keep = water[ex] <= 0
            closed[lab[ex[keep]]] = True
            water[ex[keep]] = 0.0
        f = front[has[front]]
        np.add.at(water, down[f], water[f])
    # a lake whose spill cell is reached with surplus water overflows; ones never resolved are treated as open
    end_closed = exit_cell & closed[lab]
    # every cell's terminal: follow down until the sea, a closed lake's spill cell, or nothing (pointer jumping)
    nxt = np.where(has & ~end_closed, down, np.arange(n))
    for _ in range(40):
        nn = nxt[nxt]
        if np.array_equal(nn, nxt):
            break
        nxt = nn
    endorheic = end_closed[nxt] & ~oc
    acc = water.reshape(H, W)
    return {"endorheic": endorheic.reshape(H, W), "runoff_mm": R, "pet_mm": pet, "depth": depth.reshape(H, W),
            "lakes": (lab.reshape(H, W) > 0) & closed[lab].reshape(H, W), "discharge": acc, "fdir": fdir, "area": area,
            "lake_id": lab.reshape(H, W), "n_depressions": nd, "n_sed_filled": n_sed_filled,
            "catchment_km2": catch.reshape(H, W)}

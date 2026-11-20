"""Rough annual climate for screening (Python port of the JS estimator checked against Earth's Köppen map).

Equirectangular grid, row 0 = north. Biomes: 0 rainforest 1 savanna 2 desert 3 steppe 4 temperate forest 5 boreal
6 tundra 7 ice 8 alpine. Earth check (1 degree): 3-group agreement (wet-habitable / dry / cold) ~74%.
Earth-calibrated: the planet's 22 h day shifts the dry subtropics ~1-2.5 degrees equatorward (CLAUDE.md), not modelled
here; the final climate comes from ExoPlaSim with the planet's own parameters.
"""
import math
import numpy as np
from scipy.ndimage import uniform_filter1d

TSL = [[0, 27], [10, 26.5], [20, 25], [30, 20], [40, 14], [50, 7], [60, 0], [70, -8], [80, -16], [90, -22]]
ZT = [[0, 1], [5, 1], [10, 0.6], [15, 0.2], [20, 0.05], [27, 0.03], [32, 0.12], [37, 0.35], [42, 0.55], [47, 0.65],
      [55, 0.65], [60, 0.5], [70, 0.3], [80, 0.15], [90, 0.1]]
ZM = [[0, 0.3], [8, 0.7], [15, 1], [25, 0.7], [32, 0.45], [40, 0.3], [48, 0], [90, 0]]
WE = [[0, 1], [25, 1], [32, 0.6], [40, 0.25], [60, 0.25], [65, 0.5], [90, 0.8]]
REC = [[0, 0.65], [15, 0.5], [30, 0.45], [60, 0.45], [90, 0.3]]
L0, H0, PREF, PM, LM, RF, OA, PC, LC, BOR, MAR, PTH = 1000, 2000, 2600, 1100, 400, 2000, 0.1, 1500, 9, 3, 6, 200


def tab(t, x):
    t = np.array(t, float)
    return np.interp(np.abs(x), t[:, 0], t[:, 1])


def blur(src, rad, mask=None):
    """Box blur, wrapping in longitude, clamped in latitude; with a mask only masked cells are averaged."""
    m = np.ones_like(src) if mask is None else mask.astype(float)
    v = src * m
    k = 2 * rad + 1
    num = uniform_filter1d(uniform_filter1d(v, k, axis=1, mode="wrap"), k, axis=0, mode="constant")
    den = uniform_filter1d(uniform_filter1d(m, k, axis=1, mode="wrap"), k, axis=0, mode="constant")
    out = np.where(den > 1e-9, num / np.maximum(den, 1e-12), src)
    return out


def climate_grid(elev, land, radius_km):
    """elev, land: (H, W) arrays. Returns dict T, A (seasonal half-range), P (mm/yr), B (biome, -1 at sea)."""
    H, W = elev.shape
    lat = 90 - (np.arange(H) + 0.5) * 180 / H
    cell_km = math.pi * radius_km / H
    dy = cell_km
    relief = blur(np.maximum(elev, 0), max(1, round(120 / cell_km)))
    PE = np.zeros((H, W)); PW = np.zeros((H, W)); dE = np.zeros((H, W)); dW = np.zeros((H, W))
    for y in range(H):
        la = lat[y]
        dx = max(1.0, 2 * math.pi * radius_km * math.cos(math.radians(la)) / W)
        z = float(tab(ZT, la)); rec = float(tab(REC, la)); qf = math.exp(0.04 * (float(tab(TSL, la)) - 27))
        row_land = land[y]
        sea = np.nonzero(~row_land)[0]
        start = int(sea[0]) if len(sea) else -1
        for direction, Pout, dout in ((-1, PE, dE), (1, PW, dW)):
            m = 0.3 if start < 0 else 1.0
            dist = 3000.0 if start < 0 else 0.0
            prev = 0.0
            s0 = 0 if start < 0 else start
            for k in range(2 * W):
                x = (s0 + direction * k) % W
                if not row_land[x]:
                    m += (1 - m) * 0.5; dist = 0.0; prev = 0.0
                    continue
                dist += dx
                e = relief[y, x]; dz = max(0.0, e - prev); prev = e
                lam = z * dx / L0 + dz / H0 * (OA + (1 - OA) * min(1.0, z / 0.6))
                rain = m * (1 - math.exp(-lam))
                if k >= W:
                    Pout[y, x] = PREF * qf * rain * L0 / dx
                    dout[y, x] = dist
                m -= rain * (1 - rec)
    # monsoon: distance to the ocean on the equator side
    dEq = np.zeros((H, W))
    half = H // 2
    d = np.zeros(W)
    for y in range(half - 1, -1, -1):
        d = np.where(land[y], d + dy, 0.0); dEq[y] = d
    d = np.zeros(W)
    for y in range(half, H):
        d = np.where(land[y], d + dy, 0.0); dEq[y] = d
    mix = max(1, round(300 / cell_km))
    PEs = blur(PE, mix, land); PWs = blur(PW, mix, land)
    LAT = lat[:, None] * np.ones((1, W)); AL = np.abs(LAT)
    we = tab(WE, LAT); qf = np.exp(0.04 * (tab(TSL, LAT) - 27)); zm = tab(ZM, LAT); t0 = tab(TSL, LAT)
    conv = PC * np.exp(-(LAT / LC) ** 2)
    e = np.maximum(elev, 0)
    P = we * PEs + (1 - we) * PWs + PM * zm * qf * np.exp(-dEq / LM) + conv
    dd = we * dE + (1 - we) * dW
    c = 1 - np.exp(-dd / 1500)
    cl = lambda v, a, b: np.clip(v, a, b)
    T = t0 - 6.5 * e / 1000 - 5 * c * cl((AL - 35) / 30, 0, 1) + MAR * (1 - c) * cl((AL - 40) / 20, 0, 1)
    A = (1 + 27 * cl((AL - 8) / 57, 0, 1)) * (0.25 + 0.75 * c)
    tw, tc = T + A, T - A
    pth = 20 * T + PTH
    B = np.full((H, W), 4, np.int8)
    B = np.where(tc > -3, 4, np.where(T >= BOR, 4, 5)).astype(np.int8)
    B = np.where(tc >= 18, np.where(P >= RF, 0, 1), B)
    B = np.where(P < pth, 3, B)
    B = np.where(P < pth * 0.5, 2, B)
    B = np.where(tw < 10, np.where(e > 1500, 8, 6), B)
    B = np.where(tw < 0, 7, B)
    B = np.where(land, B, -1).astype(np.int8)
    return {"T": T, "A": A, "P": P, "B": B}


def biome_shares(B, lat):
    """Area-weighted shares of each biome over land; lat: (H,) degrees."""
    w = np.cos(np.radians(lat))[:, None] * np.ones((1, B.shape[1]))
    land = B >= 0
    tot = w[land].sum()
    return [float(w[B == k].sum() / tot) for k in range(9)]

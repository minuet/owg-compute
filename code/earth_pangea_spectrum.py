"""Angular power spectrum of the real supercontinent's outline (PIPELINE principle 0, tier 2): the indicator function of
continental crust (1 on crust, 0 elsewhere) of Pangea at 200 Ma (Müller et al. 2022 continental polygons rewound, as
earth_pangea_stats.py) and of today's continents, expanded in real spherical harmonics to degree LMAX; power per degree
C_l = sum_m a_lm^2 / (2l + 1). Used to set the statistics of the planet's starting supercontinent outline (replacing the
unsourced wave amplitudes of planet_m1.outline_score) and to check it.
Output: results/earth_pangea_spectrum.json.  Usage: .venv/bin/python planet-history-sim/tools/earth_pangea_spectrum.py"""
import json, math
import numpy as np
from earth_pangea_stats import pangea_mask
from plate_checks import HERE

LMAX = 120        # half-wavelength pi R / (l + 1/2) = 166 km on the 6371 km Earth: the coastline scale the planet grid
                  # (87 km cells) can show; the 67650-point grid resolves it ((l + 1)^2 = 14641 < 67650 points)


def plm_rows(x, lmax):
    """fully normalised associated Legendre functions (orthonormal real harmonics: Y = X cos m phi, X sin m phi, with
    the sqrt(2) for m > 0), by the standard stable recurrence (Holmes & Featherstone 2002, J. Geodesy 76:279) - scipy's
    lpmv overflows above degree ~85 (2026-11-03: NaN for l >= 86). Yields (m, {l: X_lm(x)}) for m = 0..lmax."""
    st = np.sqrt(np.clip(1 - x * x, 0, None))
    pmm = np.full_like(x, math.sqrt(1 / (4 * math.pi)))
    for m in range(lmax + 1):
        if m > 0:
            pmm = -math.sqrt((2 * m + 1) / (2 * m)) * st * pmm
        rows = {m: pmm}
        if m < lmax:
            p1 = math.sqrt(2 * m + 3) * x * pmm; rows[m + 1] = p1; p0 = pmm
            for l in range(m + 2, lmax + 1):
                a = math.sqrt((4 * l * l - 1) / (l * l - m * m))
                b = math.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
                p0, p1 = p1, a * (x * p1 - b * p0)
                rows[l] = p1
        f = 1.0 if m == 0 else math.sqrt(2.0)
        yield m, {l: f * v for l, v in rows.items()}


def spectrum(g, f, lmax=LMAX):
    th = np.arccos(np.clip(g[:, 2], -1, 1)); ph = np.arctan2(g[:, 1], g[:, 0]); x = np.cos(th)
    w = 4 * math.pi / len(g)
    f = f - f.mean()                                   # degree 0 is the area share, given separately
    C = np.zeros(lmax + 1)
    for m, rows in plm_rows(x, lmax):
        cm, sm = np.cos(m * ph), np.sin(m * ph)
        for l in range(max(m, 1), lmax + 1):
            X = rows[l]
            a = (X * cm * f).sum() * w
            b = (X * sm * f).sum() * w if m else 0.0
            C[l] += a * a + b * b
    return C / (2 * np.arange(lmax + 1) + 1)


def main():
    g, crust, crust0 = pangea_mask(200.0)
    out = {"source": "Müller et al. 2022 continental polygons (200 Ma rewound / 0 Ma), indicator of crust, real "
                     "fully normalised harmonics; C_l = sum_m a_lm^2 / (2l+1)", "lmax": LMAX}
    for name, m in (("pangea_200Ma", crust), ("earth_0Ma", crust0)):
        C = spectrum(g, m.astype(float))
        out[name] = {"crust_frac": float(m.mean()), "C_l": C.tolist()}
        print(name, "share %.3f" % m.mean(), "C_l l=1..8:", " ".join("%.2e" % c for c in C[1:9]),
              " l=20,40,80,120:", " ".join("%.2e" % C[l] for l in (20, 40, 80, 120)), flush=True)
    (HERE / "results/earth_pangea_spectrum.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()

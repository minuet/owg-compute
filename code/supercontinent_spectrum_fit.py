"""Statistics of the planet's starting supercontinent outline from Pangea (PIPELINE principle 0, tier 2; 2026-11-03,
replacing the unsourced wave amplitudes and spectral slopes of planet_m1.outline_score).

The outline is an unknown initial condition, so it is random - but with the shape statistics of the real supercontinent:
the angular power spectrum C_l of Pangea's crust indicator at 200 Ma (earth_pangea_spectrum.py). The random field is a sum
of plane waves cos(f d.x + phase) with random directions d (the representation planet_m1 replays on finer grids); a wave
of frequency f puts power 2 pi A^2 j_l(f)^2 into degree l (plane-wave expansion, j_l the spherical Bessel function; checked
numerically below). Crust = the connected piece around a seed where the field is highest, of the given area (planet_m1's
priority flood). The wave amplitudes per frequency band are found by iterating A_b^2 <- A_b^2 * (target / mean measured)
over the band's degrees, averaged over N_REAL realisations, at Pangea's crust share (35.2%), until the indicator spectrum
matches Pangea's within TOL. The fitted amplitudes are a property of the random field and are used at the planet's own
crust share. Output: results/supercontinent_spectrum_fit.json.
Usage (Kaggle, ~1 h): .venv/bin/python planet-history-sim/tools/supercontinent_spectrum_fit.py"""
import json, math, heapq, time
import numpy as np
from scipy.special import spherical_jn
from earth_pangea_spectrum import plm_rows
from scipy.spatial import cKDTree
from plate_engine import fibonacci_xyz
from plate_checks import HERE

N = 60000                 # the planet's grid (planet_m1 default)
LMAX = 120                # as earth_pangea_spectrum.py
L_EDGES = [1, 2, 3, 4, 6, 9, 14, 21, 32, 48, 72, 121]   # degree bins (factor ~1.5, numerical); the spectrum is compared
                          # as total power per bin: Pangea is ONE realisation, and single degrees scatter by chance
                          # (its C_4 is 30x below its neighbours) - a random field cannot and should not copy that
F_BANDS = np.array(L_EDGES, float)   # one wave band per degree bin (a wave of frequency f puts its power near l ~ f)
WAVES_PER_BAND = 12       # random directions per band (isotropy; numerical)
N_REAL = 16               # realisations averaged per iteration (numerical)
N_ITER = 40               # at most (numerical)
DAMP = 0.5                # damped update amp2 *= (target / measured)^DAMP (numerical: the bands interact through the threshold)
TOL = 0.10                # per bin: stop when |ln(target / measured)| < max(ln 1.1, s_b), s_b = sqrt(2 / N_b) the relative
                          # scatter of the power of one realisation in a bin of N_b = sum(2l+1) modes (chi-square, Gaussian
                          # field): matching one sample closer than its own chance scatter is overfitting


def ylm_matrix(g, lmax):
    th = np.arccos(np.clip(g[:, 2], -1, 1)); ph = np.arctan2(g[:, 1], g[:, 0]); x = np.cos(th)
    rows, deg = [], []
    for m, P in plm_rows(x, lmax):
        cm, sm = np.cos(m * ph), np.sin(m * ph)
        for l in range(max(m, 1), lmax + 1):
            X = P[l]
            rows.append((X * cm).astype(np.float32)); deg.append(l)
            if m:
                rows.append((X * sm).astype(np.float32)); deg.append(l)
    return np.array(rows), np.array(deg)


def spec(Y, deg, f, lmax):
    a = (Y @ (f - f.mean()).astype(np.float32)).astype(float) * (4 * math.pi / Y.shape[1])
    return np.bincount(deg, weights=a * a, minlength=lmax + 1) / (2 * np.arange(lmax + 1) + 1)


def flood(g, nb, score, frac, seed):
    s0 = int(np.argmax(g @ seed)); seen = np.zeros(len(g), bool); seen[s0] = True
    heap = [(-score[s0], s0)]; out = np.zeros(len(g), bool); need = int(frac * len(g)); got = 0
    while heap and got < need:
        _, k = heapq.heappop(heap); out[k] = True; got += 1
        for j in nb[k]:
            if not seen[j]:
                seen[j] = True; heapq.heappush(heap, (-score[j], int(j)))
    return out


def realise(rng, g, amp):
    """field = sum over bands of WAVES_PER_BAND waves, frequency log-uniform in the band, amplitude amp[b]"""
    out = np.zeros(len(g)); waves = []
    for b, a in enumerate(amp):
        for _ in range(WAVES_PER_BAND):
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            f = float(np.exp(rng.uniform(math.log(F_BANDS[b]), math.log(F_BANDS[b + 1]))))
            ph = float(rng.uniform(0, 2 * math.pi))
            out += a * np.cos(f * (g @ d) + ph); waves.append((d.tolist(), f, float(a), ph))
    return out, waves


def band_kernel(lmax):
    """mean power per degree of one unit-amplitude wave with frequency log-uniform in each band: 2 pi j_l(f)^2"""
    l = np.arange(lmax + 1)
    K = np.zeros((len(F_BANDS) - 1, lmax + 1))
    for b in range(len(F_BANDS) - 1):
        fs = np.geomspace(F_BANDS[b], F_BANDS[b + 1], 16)
        K[b] = np.mean([2 * math.pi * spherical_jn(l, f) ** 2 for f in fs], axis=0)
    return K


def main():
    t0 = time.time()
    target = np.array(json.loads((HERE / "results/earth_pangea_spectrum.json").read_text())["pangea_200Ma"]["C_l"])
    frac = json.loads((HERE / "results/earth_pangea_spectrum.json").read_text())["pangea_200Ma"]["crust_frac"]
    g = fibonacci_xyz(N); nb = cKDTree(g).query(g, k=7)[1][:, 1:]
    Y, deg = ylm_matrix(g, LMAX)
    print(f"harmonics {Y.shape} ({time.time() - t0:.0f}s)", flush=True)
    K = band_kernel(LMAX)
    # check of the kernel on single waves (field spectrum, not thresholded)
    rng = np.random.default_rng(0)
    for f in (3.0, 20.0, 80.0):
        acc = np.zeros(LMAX + 1)
        for _ in range(20):
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            acc += spec(Y, deg, np.cos(f * (g @ d) + rng.uniform(0, 6.3)), LMAX) / 20
        th = 2 * math.pi * spherical_jn(np.arange(LMAX + 1), f) ** 2
        print(f"kernel check f={f}: measured/theory over l=1..{min(LMAX, int(1.5 * f) + 2)}: "
              f"{acc[1:int(1.5 * f) + 3].sum() / th[1:int(1.5 * f) + 3].sum():.3f}", flush=True)
    from scipy.optimize import nnls
    ll = np.arange(LMAX + 1)
    bins = [np.arange(a, b) for a, b in zip(L_EDGES[:-1], L_EDGES[1:])]
    binpow = lambda C: np.array([((2 * ll[B] + 1) * C[B]).sum() for B in bins])
    P_t = binpow(target)
    tol_b = np.maximum(math.log(1 + TOL), np.sqrt(2.0 / np.array([(2 * B + 1).sum() for B in bins])))
    # first guess: field power per bin = target, split over the wave bands by non-negative least squares on the kernel
    Kb = np.array([binpow(K[b]) for b in range(len(K))])
    amp2 = np.maximum(nnls(Kb.T, P_t)[0], 1e-6 * P_t.max())
    hist = []
    for it in range(N_ITER):
        C = np.zeros(LMAX + 1)
        for r in range(N_REAL):
            rr = np.random.default_rng(1000 * it + r)
            fld, _ = realise(rr, g, np.sqrt(amp2) / WAVES_PER_BAND ** 0.5)
            seed = g[int(np.argmax(fld))]
            C += spec(Y, deg, flood(g, nb, fld, frac, seed).astype(float), LMAX) / N_REAL
        ratio = np.log(P_t / np.maximum(binpow(C), 1e-30))
        worst = float((np.abs(ratio) / tol_b).max())          # < 1: every bin within its tolerance
        hist.append({"iter": it, "worst_over_tol": worst, "ln_ratio": ratio.tolist(), "C_l": C.tolist(), "amp2": amp2.tolist()})
        print(f"iter {it}: ln(target/measured) per bin " + " ".join(f"{r:+.2f}" for r in ratio) +
              f"  worst/tol {worst:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        if worst < 1.0:
            break
        amp2 = amp2 * np.exp(DAMP * ratio)
    out = {"source": "fit of plane-wave band amplitudes to Pangea's crust-indicator spectrum (earth_pangea_spectrum.json)",
           "F_BANDS": F_BANDS.tolist(), "WAVES_PER_BAND": WAVES_PER_BAND, "crust_frac_fit": frac,
           "amp2": amp2.tolist(), "converged": worst < 1.0, "L_EDGES": L_EDGES, "tol_per_bin": tol_b.tolist(), "history": hist, "target_C_l": target.tolist()}
    (HERE / "results/supercontinent_spectrum_fit.json").write_text(json.dumps(out))
    print("saved", flush=True)


if __name__ == "__main__":
    main()

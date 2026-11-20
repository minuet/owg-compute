"""fit_check.py on Earth (CLAUDE.md: test every method on Earth first). ETOPO5 coastline at 0.25 deg, with the Panama
and Suez isthmuses cut so the continents are separate as in the drift story. A good method finds South America-Africa
(the textbook fit) and does not find fits between continents that were never joined along those coasts.

Usage: .venv/bin/python planet-history-sim/tools/fit_earth_check.py [match_km]
"""
import sys
import numpy as np
import look_check as L
import fit_check as F
from plate_checks import HERE

F.R = L.R = 6371.0
raw = np.fromfile(HERE.parent / "reference/earth/topography/ETOPO5.DAT", dtype=">i2").reshape(2160, 4320)
e = raw.reshape(720, 3, 1440, 3).mean(axis=(1, 3))          # 5' -> 0.25 deg; row 0 at 90N, column 0 at 0E
e = np.roll(e, 720, axis=1)                                  # column 0 at 180W, like look_check's raster
land = e > 0
def cut(lat0, lat1, lon0, lon1):
    land[int((90 - lat1) * 4):int((90 - lat0) * 4), int((lon0 + 180) * 4):int((lon1 + 180) * 4)] = False
cut(7, 10, -81, -77)      # Panama
cut(28, 33, 31, 36)       # Suez and Sinai
cut(35.5, 36.5, -6, -5)   # Gibraltar
cut(12, 13.5, 42.5, 44)   # Bab-el-Mandeb
import os
if os.environ.get("SHELF"): land[:] = e > -float(os.environ["SHELF"]); cut(7, 10, -81, -77); cut(28, 33, 31, 36); cut(35.5, 36.5, -6, -5); cut(12, 13.5, 42.5, 44)
NAMES = {"Africa": (5, 20), "South America": (-10, -55), "North America": (45, -100), "Eurasia": (50, 90),
         "Australia": (-25, 135), "Antarctica": (-80, 0), "Greenland": (72, -40), "Madagascar": (-20, 46.5)}
match = float(sys.argv[1]) if len(sys.argv) > 1 else F.MATCH_KM
pairs, lab, ids, ar = F.fit_land(land, match, min_frac=0.003)
name = {lab[int((90 - la) * 4), int((lo + 180) * 4)]: n for n, (la, lo) in NAMES.items()}
for p in pairs:
    a, b = name.get(p["idA"], f"{p['A']}%"), name.get(p["idB"], f"{p['B']}%")
    print(f"{a:>14} <- {b:<14} matched {p['length_km']:5d} km  bulge {p['bulge_km']:4d}  overlap {p['overlap']:.3f}  "
          f"rms {p['rms_km']:3d}  cover {p['cover']:.2f}  gap {p['gap_km']:4d}  {'FIT' if p['fits'] else ''}")

"""scipy's cKDTree with multi-core queries by default (results are identical; only the search is split over threads).
Threads per process: env KD_THREADS (default 1 = same as plain scipy; more threads only finish sooner, the total work and heat are the same)."""
import os
from scipy.spatial import cKDTree as _cKDTree

THREADS = int(os.environ.get("KD_THREADS", "1"))


class cKDTree(_cKDTree):
    def query(self, x, k=1, eps=0, p=2, distance_upper_bound=float("inf"), workers=None):
        return super().query(x, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound,
                             workers=THREADS if workers is None else workers)

    def query_ball_point(self, x, r, p=2.0, eps=0, workers=None, **kw):
        return super().query_ball_point(x, r, p=p, eps=eps, workers=THREADS if workers is None else workers, **kw)

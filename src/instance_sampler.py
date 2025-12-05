"""Structured TSP point samplers used for generating synthetic splits.

Distributions supported:
- uniform: iid points in a bounding box.
- clustered: mixture of anisotropic Gaussians with random rotations.
- grid: jittered grid with optional dropped rows/cols to mimic corridors/obstacles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


BBox = Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class SampleResult:
    coords: np.ndarray  # shape (n, 2)
    metadata: Dict[str, object]


class StructuredSampler:
    """Sampler for structured point sets."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = _rng(seed)

    def sample_uniform(self, n: int, bbox: BBox = (0.0, 1.0, 0.0, 1.0)) -> SampleResult:
        xmin, xmax, ymin, ymax = bbox
        xs = self.rng.uniform(xmin, xmax, size=n)
        ys = self.rng.uniform(ymin, ymax, size=n)
        coords = np.stack([xs, ys], axis=1)
        meta = {"distribution": "uniform", "bbox": bbox, "n": n}
        return SampleResult(coords=coords, metadata=meta)

    def sample_clustered(
        self,
        n: int,
        k: int = 4,
        bbox: BBox = (0.0, 1.0, 0.0, 1.0),
        cov_min: float = 0.001,
        cov_max: float = 0.02,
    ) -> SampleResult:
        """Mixture of K anisotropic Gaussians."""
        xmin, xmax, ymin, ymax = bbox
        weights = self.rng.dirichlet(alpha=np.ones(k))
        counts = np.maximum(1, np.round(weights * n).astype(int))
        # Adjust to exact n
        while counts.sum() > n:
            counts[self.rng.integers(0, k)] -= 1
        while counts.sum() < n:
            counts[self.rng.integers(0, k)] += 1

        clusters = []
        for _ in range(k):
            mean = np.array(
                [self.rng.uniform(xmin, xmax), self.rng.uniform(ymin, ymax)]
            )
            # Random anisotropic covariance via rotation + eigenvalues
            theta = self.rng.uniform(0, 2 * math.pi)
            c, s = math.cos(theta), math.sin(theta)
            rot = np.array([[c, -s], [s, c]])
            eigs = self.rng.uniform(cov_min, cov_max, size=2)
            cov = rot @ np.diag(eigs) @ rot.T
            clusters.append((mean, cov))

        pts = []
        labels = []
        for idx, (cnt, (mean, cov)) in enumerate(zip(counts, clusters)):
            pts.append(self.rng.multivariate_normal(mean, cov, size=cnt))
            labels.extend([idx] * cnt)
        coords = np.vstack(pts)
        meta = {
            "distribution": "clustered",
            "bbox": bbox,
            "n": n,
            "k": k,
            "cov_min": cov_min,
            "cov_max": cov_max,
            "cluster_counts": counts.tolist(),
            "labels": labels,
        }
        return SampleResult(coords=coords, metadata=meta)

    def sample_grid(
        self,
        n: int,
        bbox: BBox = (0.0, 1.0, 0.0, 1.0),
        jitter: float = 0.01,
        drop_prob: float = 0.0,
    ) -> SampleResult:
        """Jittered grid with optional dropped rows/cols to create corridors."""
        xmin, xmax, ymin, ymax = bbox
        # Choose grid dims close to square, enough points
        rows = int(math.sqrt(n))
        cols = int(math.ceil(n / rows))
        xs = np.linspace(xmin, xmax, num=cols)
        ys = np.linspace(ymin, ymax, num=rows)
        xv, yv = np.meshgrid(xs, ys)
        base = np.stack([xv.ravel(), yv.ravel()], axis=1)

        # Optionally drop some rows/cols
        if drop_prob > 0.0:
            keep_rows = [i for i in range(rows) if self.rng.random() > drop_prob]
            keep_cols = [j for j in range(cols) if self.rng.random() > drop_prob]
            if keep_rows and keep_cols:
                mask = np.zeros((rows, cols), dtype=bool)
                for i in keep_rows:
                    for j in keep_cols:
                        mask[i, j] = True
                base = base[mask.ravel()]

        # Limit to n points
        if base.shape[0] > n:
            idx = self.rng.choice(base.shape[0], size=n, replace=False)
            base = base[idx]

        # Jitter
        noise = self.rng.normal(scale=jitter, size=base.shape)
        coords = base + noise

        meta = {
            "distribution": "grid",
            "bbox": bbox,
            "n": n,
            "rows": rows,
            "cols": cols,
            "jitter": jitter,
            "drop_prob": drop_prob,
        }
        return SampleResult(coords=coords, metadata=meta)

    def sample(self, kind: str, n: int, **kwargs) -> SampleResult:
        if kind == "uniform":
            return self.sample_uniform(n=n, **kwargs)
        if kind == "clustered":
            return self.sample_clustered(n=n, **kwargs)
        if kind == "grid":
            return self.sample_grid(n=n, **kwargs)
        raise ValueError(f"Unsupported distribution: {kind}")

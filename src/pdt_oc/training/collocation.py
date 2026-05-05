from typing import Optional, Tuple

import numpy as np


def sample_collocation_points(
    t_span: Tuple[float, float],
    n: int,
    jitter: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=float)
    t0, t1 = t_span
    if t1 <= t0:
        raise ValueError((t0, t1))
    if jitter:
        if rng is None:
            rng = np.random.default_rng()
        edges = np.linspace(0.0, 1.0, n + 1)
        widths = np.diff(edges)
        u = edges[:-1] + widths * rng.uniform(0.0, 1.0, size=n)
    else:
        u = (np.arange(n, dtype=float) + 0.5) / n
    return t0 + u * (t1 - t0)

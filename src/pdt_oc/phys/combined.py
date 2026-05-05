from typing import Callable

import numpy as np

from .gompertz import dV_dt
from .hill import effect
from .state import Idx, PhysioParams, STATE_DIM
from .two_compartment import dC_dt


def f_physics(
    t: float,
    s: np.ndarray,
    params: PhysioParams,
    dose_fn: Callable[[float], float],
) -> np.ndarray:
    V = float(s[Idx.V])
    C1 = float(s[Idx.C1])
    C2 = float(s[Idx.C2])
    growth = dV_dt(V, params.alpha, params.K)
    kill = effect(C1, params) * V
    dC1, dC2 = dC_dt(C1, C2, dose_fn(t), params)
    out = np.empty(STATE_DIM, dtype=float)
    out[Idx.V] = growth - kill
    out[Idx.C1] = dC1
    out[Idx.C2] = dC2
    return out

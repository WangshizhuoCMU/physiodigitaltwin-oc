import numpy as np

from .state import PhysioParams


def effect(C1: float, params: PhysioParams) -> float:
    if C1 <= 0.0:
        return 0.0
    cg = C1 ** params.gamma
    return params.Emax * cg / (params.EC50 ** params.gamma + cg)


def effect_trace(C1_trace: np.ndarray, params: PhysioParams) -> np.ndarray:
    arr = np.maximum(np.asarray(C1_trace, dtype=float), 0.0)
    cg = np.power(arr, params.gamma)
    return params.Emax * cg / (params.EC50 ** params.gamma + cg)

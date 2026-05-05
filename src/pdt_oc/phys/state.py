from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Idx(IntEnum):
    V = 0
    C1 = 1
    C2 = 2


STATE_DIM = len(Idx)


@dataclass(frozen=True)
class PhysioParams:
    alpha: float
    K: float
    CL: float
    Vc: float
    Vp: float
    Q: float
    Emax: float
    EC50: float
    gamma: float


def carboplatin_defaults() -> PhysioParams:
    return PhysioParams(
        alpha=0.30,
        K=100.0,
        CL=7.38,
        Vc=17.1,
        Vp=411.0,
        Q=22.0,
        Emax=1.0,
        EC50=15.0,
        gamma=2.0,
    )


def make_state(V: float, C1: float = 0.0, C2: float = 0.0) -> np.ndarray:
    s = np.zeros(STATE_DIM, dtype=float)
    s[Idx.V] = V
    s[Idx.C1] = C1
    s[Idx.C2] = C2
    return s

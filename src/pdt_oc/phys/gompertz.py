import math


def dV_dt(V: float, alpha: float, K: float) -> float:
    if V <= 0.0:
        return 0.0
    return -alpha * V * math.log(V / K)

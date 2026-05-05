from typing import Callable, Sequence

import numpy as np


def constant_infusion(rate: float, t_start: float, t_end: float) -> Callable[[float], float]:
    def dose_fn(t: float) -> float:
        return rate if t_start <= t < t_end else 0.0

    return dose_fn


def bolus_schedule(
    times: Sequence[float],
    doses: Sequence[float],
    infusion_minutes: float = 30.0,
) -> Callable[[float], float]:
    times_arr = np.asarray(times, dtype=float)
    doses_arr = np.asarray(doses, dtype=float)
    duration = infusion_minutes / 60.0
    rates = doses_arr / duration
    half = duration / 2.0

    def dose_fn(t: float) -> float:
        active = (t >= times_arr - half) & (t < times_arr + half)
        if not np.any(active):
            return 0.0
        return float(np.sum(rates[active]))

    return dose_fn


def zero_dose() -> Callable[[float], float]:
    def dose_fn(t: float) -> float:
        return 0.0

    return dose_fn

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp


def integrate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    s0: np.ndarray,
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    max_step: Optional[float] = None,
    method: str = "RK45",
) -> Tuple[np.ndarray, np.ndarray]:
    kwargs: dict = {"method": method, "rtol": rtol, "atol": atol, "dense_output": False}
    if t_eval is not None:
        kwargs["t_eval"] = t_eval
    if max_step is not None:
        kwargs["max_step"] = max_step
    sol = solve_ivp(rhs, t_span, s0, **kwargs)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y

from .state import PhysioParams


def dC_dt(
    C1: float,
    C2: float,
    dose_rate: float,
    params: PhysioParams,
) -> tuple[float, float]:
    k_elim = params.CL / params.Vc
    k_12 = params.Q / params.Vc
    k_21 = params.Q / params.Vp
    dC1 = -(k_elim + k_12) * C1 + k_21 * C2 + dose_rate / params.Vc
    dC2 = k_12 * C1 - k_21 * C2
    return dC1, dC2

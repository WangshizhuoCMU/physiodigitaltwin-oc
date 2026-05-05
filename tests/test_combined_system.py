import numpy as np
import pytest

from pdt_oc.dosing import constant_infusion, zero_dose
from pdt_oc.numerics import integrate
from pdt_oc.phys import (
    Idx,
    PhysioParams,
    carboplatin_defaults,
    effect_trace,
    f_physics,
    make_state,
)


@pytest.fixture
def params() -> PhysioParams:
    return carboplatin_defaults()


def test_zero_dose_lets_tumor_recover_to_carrying_capacity(params: PhysioParams) -> None:
    horizon = 25.0 / params.alpha
    s0 = make_state(V=0.1 * params.K)
    dose_fn = zero_dose()

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return f_physics(t, y, params, dose_fn)

    _, traj = integrate(rhs, s0, (0.0, horizon), rtol=1e-9, atol=1e-12)
    V_terminal = float(traj[Idx.V, -1])
    assert abs(V_terminal - params.K) < 1e-3, V_terminal


def test_continuous_infusion_drives_tumor_below_carrying_capacity(params: PhysioParams) -> None:
    horizon = 200.0
    rate = 80.0
    s0 = make_state(V=params.K)
    dose_fn = constant_infusion(rate, 0.0, horizon)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return f_physics(t, y, params, dose_fn)

    times, traj = integrate(
        rhs,
        s0,
        (0.0, horizon),
        rtol=1e-7,
        atol=1e-10,
        max_step=0.5,
    )
    V_trace = traj[Idx.V]
    assert float(V_trace[-1]) < params.K, float(V_trace[-1])
    assert float(np.min(V_trace)) < 0.7 * params.K, float(np.min(V_trace))
    E_trace = effect_trace(traj[Idx.C1], params)
    assert float(np.max(E_trace)) > 0.1

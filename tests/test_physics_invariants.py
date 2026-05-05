import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from pdt_oc.dosing import constant_infusion
from pdt_oc.numerics import integrate
from pdt_oc.phys import (
    PhysioParams,
    carboplatin_defaults,
    dC_dt,
    dV_dt,
    effect,
)


@pytest.fixture
def params() -> PhysioParams:
    return carboplatin_defaults()


def test_gompertz_attracts_to_carrying_capacity(params: PhysioParams) -> None:
    horizon = 25.0 / params.alpha
    tol = 1e-3

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([dV_dt(float(y[0]), params.alpha, params.K)])

    for V0 in (0.1 * params.K, 2.0 * params.K):
        _, traj = integrate(
            rhs,
            np.array([V0]),
            (0.0, horizon),
            rtol=1e-9,
            atol=1e-12,
        )
        terminal = float(traj[0, -1])
        assert math.isfinite(terminal)
        assert abs(terminal - params.K) < tol, (V0, terminal, params.K)


def test_two_compartment_auc_equals_dose_over_clearance(params: PhysioParams) -> None:
    rate = 1.0
    t_start = 0.0
    t_end = 0.5
    dose_total = rate * (t_end - t_start)
    dose_fn = constant_infusion(rate, t_start, t_end)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        d1, d2 = dC_dt(float(y[0]), float(y[1]), dose_fn(float(t)), params)
        return np.array([d1, d2])

    horizon = 800.0
    dense = np.linspace(0.0, 5.0, 2000)
    sparse = np.linspace(5.0, horizon, 8000)[1:]
    t_eval = np.concatenate([dense, sparse])

    times, traj = integrate(
        rhs,
        np.array([0.0, 0.0]),
        (0.0, horizon),
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-13,
        max_step=0.05,
    )

    auc_numerical = float(trapezoid(traj[0], times))
    auc_analytical = dose_total / params.CL
    rel_err = abs(auc_numerical - auc_analytical) / auc_analytical
    assert rel_err < 0.02, (auc_numerical, auc_analytical, rel_err)


def test_hill_at_ec50_yields_half_emax(params: PhysioParams) -> None:
    response = effect(params.EC50, params)
    assert math.isclose(response, 0.5 * params.Emax, rel_tol=1e-12, abs_tol=1e-12)

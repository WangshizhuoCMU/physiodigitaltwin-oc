import numpy as np
from scipy.integrate import trapezoid

from pdt_oc.dosing import bolus_schedule, constant_infusion, zero_dose


def _integrate_dose(dose_fn, t_min: float, t_max: float, n_points: int = 200_000) -> float:
    grid = np.linspace(t_min, t_max, n_points)
    values = np.array([dose_fn(float(t)) for t in grid], dtype=float)
    return float(trapezoid(values, grid))


def test_constant_infusion_delivers_specified_mass() -> None:
    rate = 2.5
    t_start = 1.0
    t_end = 6.0
    expected = rate * (t_end - t_start)
    dose_fn = constant_infusion(rate, t_start, t_end)
    integrated = _integrate_dose(dose_fn, t_start - 1.5, t_end + 1.5)
    rel_err = abs(integrated - expected) / expected
    assert rel_err < 1e-3, (integrated, expected, rel_err)


def test_bolus_schedule_total_mass_matches_sum_of_doses() -> None:
    times = [0.0, 24.0, 48.0]
    doses = [100.0, 100.0, 150.0]
    expected = float(sum(doses))
    dose_fn = bolus_schedule(times, doses, infusion_minutes=30.0)
    integrated = _integrate_dose(dose_fn, -1.0, 50.0, n_points=400_000)
    rel_err = abs(integrated - expected) / expected
    assert rel_err < 5e-3, (integrated, expected, rel_err)


def test_zero_dose_returns_zero_for_any_time() -> None:
    dose_fn = zero_dose()
    for t in (-1e6, -1.0, 0.0, 1.0, 100.0, 1e6):
        assert dose_fn(float(t)) == 0.0


def test_constant_infusion_active_only_in_window() -> None:
    rate = 3.0
    t_start = 2.0
    t_end = 5.0
    dose_fn = constant_infusion(rate, t_start, t_end)
    assert dose_fn(t_start - 1e-9) == 0.0
    assert dose_fn(t_start) == rate
    assert dose_fn((t_start + t_end) / 2.0) == rate
    assert dose_fn(t_end) == 0.0
    assert dose_fn(t_end + 1.0) == 0.0

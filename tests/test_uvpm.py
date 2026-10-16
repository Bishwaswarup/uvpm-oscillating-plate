"""Validation tests for the unsteady vortex panel method."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: E402
from kinematics import update_geometry  # noqa: E402
from post_processing import harmonic_amplitude, theodorsen_function, theodorsen_lift_series  # noqa: E402
from simulate import simulate  # noqa: E402
from vortex_math import build_aic_matrix  # noqa: E402


def make_params(**over):
    cfg = {**config.PHYSICS, 'N': 40, 'dt': 0.0125, 'T_total': 10.0, 'wake_chords': 25,
           'rc_fraction': 0.05, 'epsilon': 0.5, 'cutoff_periods': 1, 'grid_spacing': 'cosine'}
    cfg.update(over)
    p = config.build_params(cfg)
    return p, config.get_grid(p)


def steady_geometry(alpha):
    def fn(t, params, grid):
        g = update_geometry(0.0, {**params, 'h_amp': 0.0, 'alpha_amp': 0.0}, grid)
        ca, sa = np.cos(alpha), np.sin(alpha)
        g.update(
            X_v=grid['x_v_local'] * ca, Z_v=-grid['x_v_local'] * sa,
            X_c=grid['x_c_local'] * ca, Z_c=-grid['x_c_local'] * sa,
            X_TE=params['c'] * ca, Z_TE=-params['c'] * sa,
            n_X=np.full(params['N'], -sa), n_Z=np.full(params['N'], ca), alpha=alpha,
        )
        return g
    return fn


@pytest.mark.parametrize("grid_spacing", ["cosine", "uniform"])
def test_steady_flat_plate_lift_slope(grid_spacing):
    """A flat plate at fixed small angle must settle to C_L = 2πα."""
    p, g = make_params(grid_spacing=grid_spacing)
    alpha = 0.02
    res = simulate(p, g, geometry_fn=steady_geometry(alpha), verbose=False)
    assert abs(res['Cl'][-1] / (2 * np.pi * alpha) - 1) < 0.005


def test_steady_quarter_chord_moment():
    """Thin-airfoil theory: C_M,LE = −C_L / 4 for a flat plate."""
    p, g = make_params()
    res = simulate(p, g, geometry_fn=steady_geometry(0.02), verbose=False)
    assert abs(res['Cm'][-1] / (-res['Cl'][-1] / 4) - 1) < 0.01


def test_kelvin_total_circulation_zero():
    # wake long enough that nothing is truncated: bound + wake circulation stays 0
    p, g = make_params(T_total=5.0, wake_chords=400)
    res = simulate(p, g, verbose=False)
    bound_scale = np.pi * p['U_inf'] * p['c'] * p['alpha_amp']     # steady Γ at α_amp
    assert np.max(np.abs(res['circulation_total'])) < 1e-12 * max(bound_scale, 1.0) * 1e3


def test_core_must_not_enter_bound_aic():
    """A finite core in the bound-vortex AIC inflates lift (the v1 bug)."""
    p, g = make_params()
    geo = steady_geometry(0.02)(0.0, p, g)
    rhs = -p['U_inf'] * geo['n_X']
    gam0 = np.linalg.solve(build_aic_matrix(geo, rc=0.0), rhs)
    gam5 = np.linalg.solve(build_aic_matrix(geo, rc=0.05), rhs)
    assert gam5.sum() / gam0.sum() > 1.1


def test_theodorsen_function_limits():
    assert theodorsen_function(0.0) == 1.0
    assert abs(theodorsen_function(1e6) - 0.5) < 1e-3   # C(k → ∞) → 1/2
    c = theodorsen_function(0.1)                          # tabulated: 0.832 − 0.172i
    assert abs(c.real - 0.832) < 0.01 and abs(c.imag + 0.172) < 0.01


def test_oscillating_plate_matches_theodorsen():
    p, g = make_params(T_total=40.0, N=30)
    res = simulate(p, g, verbose=False)
    t = res['time']
    m = t >= 20.0
    alpha = np.radians(res['alpha_deg'])
    cl_th, _, _ = theodorsen_lift_series(alpha[m], res['h'][m], t[m], p['omega'], p['c'], p['U_inf'])
    a_num = harmonic_amplitude(res['Cl'][m], t[m], p['omega'])
    a_th = harmonic_amplitude(cl_th, t[m], p['omega'])
    assert abs(abs(a_num) / abs(a_th) - 1) < 0.03
    assert abs(np.degrees(np.angle(a_num / a_th))) < 2.0

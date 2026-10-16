"""Time-marching UVPM loop, shared by main.py and the tests."""
import numpy as np

from forces import calculate_forces
from kinematics import update_geometry
from solver import build_rhs_array, solve_bound_circulation
from vortex_math import build_aic_matrix
from wake_management import convect_wake, shed_new_vortex


def simulate(params, grid, geometry_fn=update_geometry, verbose=True):
    """Run the unsteady vortex panel method from rest.

    Parameters
    ----------
    params, grid : from config.get_params() / config.get_grid()
    geometry_fn  : callable (t, params, grid) -> geometry dict; defaults to the
                   prescribed heave/pitch kinematics
    verbose      : print progress every 100 steps

    Returns
    -------
    dict with arrays 'time', 'Cl', 'Cd', 'Cm', 'alpha_deg', 'h' and the final
    state 'wake_X', 'wake_Z', 'wake_Gamma', 'geometry', 'Gamma_bound',
    'Gamma_bound_prev', plus 'circulation_total' (bound + wake, per step,
    before far-wake truncation; Kelvin's theorem requires it to stay 0).
    """
    N = params['N']
    wake_X, wake_Z, wake_Gamma = np.array([]), np.array([]), np.array([])
    Gamma_bound_prev = np.zeros(N)
    out = {k: [] for k in ('Cl', 'Cd', 'Cm', 'alpha_deg', 'h', 'circulation_total')}
    Gamma_bound_curr = Gamma_bound_prev
    geometry = None

    for step in range(params['N_steps']):
        t = step * params['dt']
        geometry = geometry_fn(t, params, grid)

        # Convect existing wake with the previous bound circulation
        wake_X, wake_Z = convect_wake(wake_X, wake_Z, wake_Gamma, geometry, Gamma_bound_prev, params)

        # Bound circulation from the no-penetration condition (no core in the AIC)
        AIC = build_aic_matrix(geometry, rc=0.0)
        RHS = build_rhs_array(geometry, params, wake_X, wake_Z, wake_Gamma)
        Gamma_bound_curr = solve_bound_circulation(AIC, RHS)

        # Shed a new wake vortex (Kelvin's theorem)
        x_new, z_new, g_new = shed_new_vortex(Gamma_bound_prev, Gamma_bound_curr, geometry, params)
        wake_X = np.append(wake_X, x_new)
        wake_Z = np.append(wake_Z, z_new)
        wake_Gamma = np.append(wake_Gamma, g_new)
        out['circulation_total'].append(np.sum(Gamma_bound_curr) + np.sum(wake_Gamma))

        # Truncate the far wake to keep the cost bounded
        keep = wake_X < geometry['X_TE'] + params['wake_chords'] * params['c']
        wake_X, wake_Z, wake_Gamma = wake_X[keep], wake_Z[keep], wake_Gamma[keep]

        Cl, Cd, Cm = calculate_forces(Gamma_bound_curr, Gamma_bound_prev, geometry, params, grid)
        if verbose and step % 100 == 0:
            print(f"Step {step:5d} | t={t:7.2f}s | Wake vortices: {len(wake_Gamma):4d} | Cl={Cl:.5f}")

        out['Cl'].append(Cl)
        out['Cd'].append(Cd)
        out['Cm'].append(Cm)
        out['alpha_deg'].append(np.degrees(geometry['alpha']))
        out['h'].append(geometry['h'])
        if step < params['N_steps'] - 1:
            Gamma_bound_prev = Gamma_bound_curr.copy()

    res = {k: np.asarray(v) for k, v in out.items()}
    res.update(
        time=np.arange(params['N_steps']) * params['dt'],
        wake_X=wake_X, wake_Z=wake_Z, wake_Gamma=wake_Gamma, geometry=geometry,
        Gamma_bound=Gamma_bound_curr, Gamma_bound_prev=Gamma_bound_prev,
    )
    return res

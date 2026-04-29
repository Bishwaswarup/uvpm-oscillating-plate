import numpy as np


def get_kinematics(t, params):
    omega = params['omega']
    h_amp = params['h_amp']
    alpha_amp = params['alpha_amp']
    alpha_phase = params['alpha_phase']

    h = h_amp * np.sin(omega * t)
    h_dot = h_amp * omega * np.cos(omega * t)

    alpha = alpha_amp * np.sin(omega * t + alpha_phase)
    alpha_dot = alpha_amp * omega * np.cos(omega * t + alpha_phase)

    return h, h_dot, alpha, alpha_dot


def update_geometry(t, params, grid):
    h, h_dot, alpha, alpha_dot = get_kinematics(t, params)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # Global Coordinates
    X_v = grid['x_v_local'] * ca
    Z_v = h - grid['x_v_local'] * sa

    X_c = grid['x_c_local'] * ca
    Z_c = h - grid['x_c_local'] * sa

    # Trailing Edge Global Kinematics
    X_TE = params['c'] * ca
    Z_TE = h - params['c'] * sa
    X_dot_TE = -params['c'] * alpha_dot * sa  # Exact horizontal velocity of TE
    Z_dot_TE = h_dot - params['c'] * alpha_dot * ca  # Exact vertical velocity of TE

    # Global Unit Normal Vector
    n_X = np.full(params['N'], -sa)
    n_Z = np.full(params['N'], ca)

    # Kinematic velocity of the solid plate at the collocation points
    V_plate_X = -grid['x_c_local'] * alpha_dot * sa
    V_plate_Z = h_dot - grid['x_c_local'] * alpha_dot * ca

    geometry = {
        'X_v': X_v, 'Z_v': Z_v,
        'X_c': X_c, 'Z_c': Z_c,
        'X_TE': X_TE, 'Z_TE': Z_TE,
        'X_dot_TE': X_dot_TE, 'Z_dot_TE': Z_dot_TE,
        'n_X': n_X, 'n_Z': n_Z,
        'V_plate_X': V_plate_X, 'V_plate_Z': V_plate_Z,
        'alpha': alpha, 'h': h
    }

    return geometry

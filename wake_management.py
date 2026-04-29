import numpy as np
def convect_wake(wake_X, wake_Z, wake_Gamma, geometry, Gamma_bound_prev, params):
    M = len(wake_Gamma)
    if M == 0:
        return wake_X, wake_Z  # Nothing to move at t=0

    Xv, Zv = geometry['X_v'], geometry['Z_v']
    rc = params['rc']
    dt = params['dt']

    # 1. Influence of BOUND vortices (from previous step) on WAKE vortices
    rx_bw = wake_X[:, None] - Xv[None, :]
    rz_bw = wake_Z[:, None] - Zv[None, :]
    denom_bw = rx_bw ** 2 + rz_bw ** 2 + rc ** 2

    u_bw = Gamma_bound_prev[None, :] * rz_bw / (2 * np.pi * denom_bw)
    w_bw = Gamma_bound_prev[None, :] * (-rx_bw) / (2 * np.pi * denom_bw)

    U_bound_on_wake = u_bw.sum(axis=1)
    W_bound_on_wake = w_bw.sum(axis=1)

    # 2. Influence of OTHER WAKE vortices on WAKE vortices (Wake-on-Wake)
    rx_ww = wake_X[:, None] - wake_X[None, :]
    rz_ww = wake_Z[:, None] - wake_Z[None, :]
    denom_ww = rx_ww ** 2 + rz_ww ** 2 + rc ** 2

    u_ww = wake_Gamma[None, :] * rz_ww / (2 * np.pi * denom_ww)
    w_ww = wake_Gamma[None, :] * (-rx_ww) / (2 * np.pi * denom_ww)

    # Explicitly zero the diagonal to prevent self-induction artifacts
    np.fill_diagonal(u_ww, 0.0)
    np.fill_diagonal(w_ww, 0.0)

    U_wake_on_wake = u_ww.sum(axis=1)
    W_wake_on_wake = w_ww.sum(axis=1)

    # 3. Total Velocity at each wake vortex
    U_total = params['U_inf'] + U_bound_on_wake + U_wake_on_wake
    W_total = 0.0 + W_bound_on_wake + W_wake_on_wake

    # 4. Forward Euler Update
    new_wake_X = wake_X + U_total * dt
    new_wake_Z = wake_Z + W_total * dt

    return new_wake_X, new_wake_Z

def shed_new_vortex(Gamma_bound_prev, Gamma_bound_curr, geometry, params):
    """
    Calculates strength and initial position of the newly shed wake vortex.
    Uses exact TE kinematics (X_dot_TE and Z_dot_TE).
    """
    # 1. Strength based on Kelvin's Theorem
    Gamma_new = np.sum(Gamma_bound_prev) - np.sum(Gamma_bound_curr)

    # 2. Placement: downstream of the moving Trailing Edge
    epsilon = params['epsilon']
    dt = params['dt']

    X_new = geometry['X_TE'] + (params['U_inf'] + geometry['X_dot_TE']) * dt * epsilon
    Z_new = geometry['Z_TE'] + geometry['Z_dot_TE'] * dt * epsilon

    return X_new, Z_new, Gamma_new
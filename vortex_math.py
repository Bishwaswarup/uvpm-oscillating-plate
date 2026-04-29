import numpy as np

def biot_savart_2d(target_x, target_z, vortex_x, vortex_z, gamma, rc):
    rx = target_x - vortex_x
    rz = target_z - vortex_z

    r2 = rx ** 2 + rz ** 2
    denominator = r2 + rc**2

    u_ind = (gamma / (2 * np.pi)) * (rz / denominator)
    w_ind = (gamma / (2 * np.pi)) * (-rx / denominator)

    return u_ind, w_ind

def build_aic_matrix(geometry, rc):
    Xc, Zc = geometry['X_c'], geometry['Z_c']
    Xv, Zv = geometry['X_v'], geometry['Z_v']
    nX, nZ = geometry['n_X'], geometry['n_Z']

    # broadcasting (N,1) - (1,N) to (N,N) matrix
    rx = Xc[:, None] - Xv[None, :]
    rz = Zc[:, None] - Zv[None, :]

    denom = rx ** 2 + rz ** 2 + rc ** 2

    u_unit = rz / (2 * np.pi * denom)
    w_unit = -rx / (2 * np.pi * denom)

    A = nX[:, None] * u_unit + nZ[:, None] * w_unit
    return A

def calc_wake_induced_velocity(geometry, wake_X, wake_Z, wake_Gamma, rc):
    if len(wake_Gamma) == 0:
        N = len(geometry['X_c'])
        return np.zeros(N), np.zeros(N)

    Xc, Zc = geometry['X_c'][:, None], geometry['Z_c'][:, None]

    rx = Xc - wake_X[None, :]
    rz = Zc - wake_Z[None, :]

    denom = rx ** 2 + rz ** 2 + rc ** 2

    u_mat = wake_Gamma[None, :] * rz / (2 * np.pi * denom)
    w_mat = wake_Gamma[None, :] * (-rx) / (2 * np.pi * denom)

    U_wake_total = u_mat.sum(axis=1)
    W_wake_total = w_mat.sum(axis=1)

    return U_wake_total, W_wake_total




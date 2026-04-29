import numpy as np
from  vortex_math import calc_wake_induced_velocity

def build_rhs_array(geometry, params, wake_X, wake_Z, wake_Gamma):
    U_wake, W_wake = calc_wake_induced_velocity(
        geometry, wake_X, wake_Z, wake_Gamma, params['rc']
    )

    V_plate_X = geometry['V_plate_X']
    V_plate_Z = geometry['V_plate_Z']
    nX = geometry['n_X']
    nZ = geometry['n_Z']
    U_inf = params['U_inf']

    freestream_normal = U_inf * nX + 0.0 * nZ
    wake_normal = U_wake * nX + W_wake * nZ
    plate_normal = V_plate_X * nX + V_plate_Z * nZ

    RHS = plate_normal - freestream_normal - wake_normal
    return RHS

def solve_bound_circulation(AIC_matrix, RHS):
    """
        Solves the linear system [AIC] * {Gamma} = {RHS} to find the bound circulation.

        Note on Kutta/Kelvin Condition:
        This solver uses an explicit N x N approach. The bound circulation is solved
        first using the historical wake. The new wake vortex is shed afterward in the
        time loop based on d(Gamma)/dt, rather than simultaneously in an (N+1) matrix.
        """

    Gamma_bound = np.linalg.solve(AIC_matrix, RHS)
    return Gamma_bound

import numpy as np


def calculate_pressure_distribution(Gamma_bound_curr, Gamma_bound_prev, params, grid):
    rho = params['rho']

    # This solver uses the opposite sign for vortex circulation from the
    # aerodynamic sign convention used in Cl/Cm, so convert once here.
    Gamma_sum_curr = np.cumsum(Gamma_bound_curr)
    Gamma_sum_prev = np.cumsum(Gamma_bound_prev)

    dGamma_dt = -(Gamma_sum_curr - Gamma_sum_prev) / params['dt']

    circulatory_term = -params['U_inf'] * (Gamma_bound_curr / grid['dx'])
    Delta_P = rho * (circulatory_term + dGamma_dt)
    q_inf = 0.5 * rho * params['U_inf'] ** 2
    Cp_jump = Delta_P / q_inf
    return Delta_P, Cp_jump


def calculate_forces(Gamma_bound_curr, Gamma_bound_prev, geometry, params, grid):
    rho = params['rho']
    Delta_P, _ = calculate_pressure_distribution(
        Gamma_bound_curr, Gamma_bound_prev, params, grid
    )

    # 3. Integrate to find Normal Force and Leading Edge Moment
    F_N = np.sum(Delta_P * grid['dx'])

    # Moment uses the local, unpitched coordinate as the moment arm.
    # Note: Evaluated in the plate frame. Wind tunnel frame correction is negligible O(alpha^2).
    M_LE = -np.sum(Delta_P * grid['dx'] * grid['x_v_local'])

    # 4. Resolve Normal Force into Lift and Drag
    alpha = geometry['alpha']
    L_prime = F_N * np.cos(alpha)

    # Drag calculated here is Pressure Drag only.
    # Explicitly ignores leading-edge suction.
    D_prime = F_N * np.sin(alpha)

    # 5. Non-dimensional Coefficients
    q_inf = 0.5 * rho * params['U_inf'] ** 2
    c = params['c']

    Cl = L_prime / (q_inf * c)
    Cd = D_prime / (q_inf * c)
    Cm = M_LE / (q_inf * c ** 2)

    return Cl, Cd, Cm

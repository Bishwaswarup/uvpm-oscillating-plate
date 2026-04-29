import os

import numpy as np

from setup import configure_runtime_environment, get_params, get_grid, write_snapshot
from kinematics import get_kinematics, update_geometry
from vortex_math import build_aic_matrix
from solver import build_rhs_array, solve_bound_circulation
from wake_management import convect_wake, shed_new_vortex
from forces import calculate_forces, calculate_pressure_distribution

configure_runtime_environment()

from post_processing import (
    save_cp_distribution_plot,
    save_effective_alpha_sketch,
    save_kinematics_plot,
    save_motion_phase_schematic,
    save_panel_schematic,
    save_peak_values_table,
    save_theodorsen_comparison_plot,
    theodorsen_function,
    theodorsen_lift_series,
)

import matplotlib.pyplot as plt

params = get_params()
grid   = get_grid(params)
write_snapshot(params)

print(f"Starting simulation: preset='{params['_preset']}' | "
      f"N={params['N']} | dt={params['dt']} | "
      f"N_steps={params['N_steps']} | max_wake≈{params['_M_max_wake']}")

# Storage
Cl_history, Cd_history, Cm_history = [], [], []
alpha_history, h_history = [], []

# Wake arrays
wake_X     = np.array([])
wake_Z     = np.array([])
wake_Gamma = np.array([])

# Previous bound circulation (zero → starts from rest)
Gamma_bound_prev = np.zeros(params['N'])
Gamma_bound_final = None
Gamma_bound_final_prev = None

# 2. Main Time-Stepping Loop
for step in range(params['N_steps']):
    t = step * params['dt']

    # A. Update geometry & kinematics
    geometry = update_geometry(t, params, grid)

    # B. Convect existing wake using PREVIOUS bound circulation
    wake_X, wake_Z = convect_wake(
        wake_X, wake_Z, wake_Gamma, geometry, Gamma_bound_prev, params
    )

    # C. Build AIC matrix and RHS (wake influence included)
    AIC_matrix = build_aic_matrix(geometry, params['rc'])
    RHS        = build_rhs_array(geometry, params, wake_X, wake_Z, wake_Gamma)

    # D. Solve for bound circulation
    Gamma_bound_curr = solve_bound_circulation(AIC_matrix, RHS)

    # E. Shed new wake vortex (Kelvin's theorem)
    x_new, z_new, g_new = shed_new_vortex(
        Gamma_bound_prev, Gamma_bound_curr, geometry, params
    )

    # F. Append new vortex
    wake_X     = np.append(wake_X,     x_new)
    wake_Z     = np.append(wake_Z,     z_new)
    wake_Gamma = np.append(wake_Gamma, g_new)

    # F2. Truncate far wake — keeps M bounded
    far_wake_limit = geometry['X_TE'] + params['wake_chords'] * params['c']
    keep       = wake_X < far_wake_limit
    wake_X     = wake_X[keep]
    wake_Z     = wake_Z[keep]
    wake_Gamma = wake_Gamma[keep]

    # G. Calculate unsteady forces
    Cl, Cd, Cm = calculate_forces(
        Gamma_bound_curr, Gamma_bound_prev, geometry, params, grid
    )

    # Diagnostic print every 100 steps
    if step % 100 == 0:
        print(f"Step {step:4d} | t={t:6.2f}s | "
              f"Wake vortices: {len(wake_Gamma):3d} | Cl={Cl:.4f}")

    # H. Store data
    Cl_history.append(Cl)
    Cd_history.append(Cd)
    Cm_history.append(Cm)
    alpha_history.append(np.degrees(geometry['alpha']))
    h_history.append(geometry['h'])

    if step == params['N_steps'] - 1:
        Gamma_bound_final = Gamma_bound_curr.copy()
        Gamma_bound_final_prev = Gamma_bound_prev.copy()

    Gamma_bound_prev = Gamma_bound_curr.copy()

print("Simulation complete.")

# 3. Post-processing
alpha_deg = np.array(alpha_history)
h_arr      = np.array(h_history)
Cl_arr    = np.array(Cl_history)
Cd_arr    = np.array(Cd_history)
Cm_arr    = np.array(Cm_history)
time_arr  = np.arange(params['N_steps']) * params['dt']

alpha_rad_arr = np.zeros_like(time_arr)
for i, t in enumerate(time_arr):
    _, _, alpha_rad_arr[i], _ = get_kinematics(t, params)

# Discard startup transient (set in setup.py via cutoff_periods)
cutoff     = params['cutoff']
alpha_clean = alpha_deg[cutoff:]
Cl_clean    = Cl_arr[cutoff:]
Cd_clean    = Cd_arr[cutoff:]
Cm_clean    = Cm_arr[cutoff:]
time_clean  = time_arr[cutoff:]


def extrema_row(name, values, time_values):
    min_idx = int(np.argmin(values))
    max_idx = int(np.argmax(values))
    return [
        name,
        values[min_idx],
        time_values[min_idx],
        values[max_idx],
        time_values[max_idx],
    ]

# 4. Plotting

# PLOT 1: Lift time history
fig_cl, ax_cl = plt.subplots(figsize=(10, 4))
ax_cl.plot(time_clean, Cl_clean, 'b')
ax_cl.set_title(
    f'Lift Coefficient History  '
    f'[N={params["N"]}, dt={params["dt"]}, k={params["_k_reduced"]:.4f}]',
    fontsize=13
)
ax_cl.set_ylabel('$C_L$')
ax_cl.set_xlabel('Time (s)')
ax_cl.grid(True)
plt.tight_layout()
plt.savefig('cl_history.png', dpi=300)
print("Saved → cl_history.png")

# PLOT 2: Drag time history
fig_cd, ax_cd = plt.subplots(figsize=(10, 4))
ax_cd.plot(time_clean, Cd_clean, 'r')
ax_cd.set_title(
    f'Pressure Drag History  '
    f'[N={params["N"]}, dt={params["dt"]}, k={params["_k_reduced"]:.4f}]',
    fontsize=13
)
ax_cd.set_ylabel('$C_D$')
ax_cd.set_xlabel('Time (s)')
ax_cd.grid(True)
plt.tight_layout()
plt.savefig('cd_history.png', dpi=300)
print("Saved → cd_history.png")

# PLOT 3: Moment time history
fig_cm, ax_cm = plt.subplots(figsize=(10, 4))
ax_cm.plot(time_clean, Cm_clean, 'g')
ax_cm.set_title(
    f'Leading-Edge Moment History  '
    f'[N={params["N"]}, dt={params["dt"]}, k={params["_k_reduced"]:.4f}]',
    fontsize=13
)
ax_cm.set_ylabel('$C_M$')
ax_cm.set_xlabel('Time (s)')
ax_cm.grid(True)
plt.tight_layout()
plt.savefig('cm_history.png', dpi=300)
print("Saved → cm_history.png")

# PLOT 4: Hysteresis
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle(
    f'Aerodynamic Hysteresis  [k={params["_k_reduced"]:.4f}, '
    f'narrow ellipse expected for quasi-steady regime]',
    fontsize=12
)

ax2[0].plot(alpha_clean, Cl_clean, 'b')
# Add quasi-steady reference line
alpha_deg_range = np.linspace(alpha_clean.min(), alpha_clean.max(), 100)
Cl_qs = 2 * np.pi * np.deg2rad(alpha_deg_range)
ax2[0].plot(alpha_deg_range, Cl_qs, 'b--', alpha=0.4, label='Quasi-steady $2\\pi\\alpha$')
ax2[0].set_title('Lift Coefficient vs $\\alpha$')
ax2[0].set_xlabel('Angle of Attack (deg)')
ax2[0].set_ylabel('$C_L$')
ax2[0].legend()
ax2[0].grid(True)

ax2[1].plot(alpha_clean, Cm_clean, 'g')
ax2[1].set_title('Pitching Moment vs $\\alpha$')
ax2[1].set_xlabel('Angle of Attack (deg)')
ax2[1].set_ylabel('$C_M$')
ax2[1].grid(True)

plt.tight_layout()
plt.savefig('alpha_hysteresis.png', dpi=300)
print("Saved → alpha_hysteresis.png")

# PLOT 5: Wake geometry
plt.figure(figsize=(12, 3))
plt.title(
    f'Free-Vortex Wake at Final Time Step  '
    f'[{len(wake_Gamma)} vortices retained, {params["wake_chords"]}c limit]',
    fontsize=12
)
alpha_final = geometry['alpha']
LE_X = geometry['X_v'][0] - 0.25 * grid['dx'][0] * np.cos(alpha_final)
LE_Z = geometry['Z_v'][0] + 0.25 * grid['dx'][0] * np.sin(alpha_final)
plt.plot([LE_X, geometry['X_TE']], [LE_Z, geometry['Z_TE']],
         'k-', linewidth=4, label='Flat Plate')

# Colour wake vortices by strength
sc = plt.scatter(wake_X, wake_Z, s=8,
                 c=wake_Gamma, cmap='RdBu', alpha=0.7,
                 label='Shed Vortices')
plt.colorbar(sc, label='$\\Gamma$ (m²/s)')
plt.xlabel('Global X (m)')
plt.ylabel('Global Z (m)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig('wake_after_plate.png', dpi=300)
print("Saved → wake_after_plate.png")

save_kinematics_plot(time_arr, h_arr, alpha_deg, params, 'kinematics.png')
print("Saved → kinematics.png")

phase_times = [0.0, 0.25 * params['_T_osc'], 0.5 * params['_T_osc'], 0.75 * params['_T_osc'], params['_T_osc']]
phase_geometries = [update_geometry(t_phase, params, grid) for t_phase in phase_times]
save_motion_phase_schematic(phase_times, phase_geometries, params, 'motion_phase_schematic.png')
print("Saved → motion_phase_schematic.png")

schematic_geometry = update_geometry(0.0, params, grid)
save_panel_schematic(params, grid, schematic_geometry, 'panel_schematic.png')
print("Saved → panel_schematic.png")

h_effective, h_dot_effective, alpha_effective, _ = get_kinematics(0.0, params)
effective_geometry = update_geometry(0.0, params, grid)
save_effective_alpha_sketch(
    effective_geometry,
    h_dot_effective,
    params,
    'effective_alpha_sketch.png',
)
print("Saved → effective_alpha_sketch.png")

_, Cp_jump_final = calculate_pressure_distribution(
    Gamma_bound_final, Gamma_bound_final_prev, params, grid
)
save_cp_distribution_plot(
    grid['x_c_local'] / params['c'],
    Cp_jump_final,
    time_arr[-1],
    'cp_distribution.png',
)
print("Saved → cp_distribution.png")

cycle_steps = max(2, int(round(params['_T_osc'] / params['dt'])))
cycle_start = max(cutoff, params['N_steps'] - cycle_steps)
time_cycle = time_arr[cycle_start:]
cl_cycle = Cl_arr[cycle_start:]
cl_theodorsen_cycle, cl_quasi_steady_cycle, c_k = theodorsen_lift_series(
    alpha_rad_arr[cycle_start:],
    h_arr[cycle_start:],
    time_cycle,
    params['omega'],
    params['c'],
    params['U_inf'],
)
save_theodorsen_comparison_plot(
    time_cycle,
    cl_cycle,
    cl_quasi_steady_cycle,
    cl_theodorsen_cycle,
    c_k,
    params,
    'theodorsen_comparison.png',
)
print("Saved → theodorsen_comparison.png")

peak_rows = [
    extrema_row('h_m', h_arr, time_arr),
    extrema_row('alpha_deg', alpha_deg, time_arr),
    extrema_row('Cl', Cl_clean, time_clean),
    extrema_row('Cd_pressure', Cd_clean, time_clean),
    extrema_row('Cm_le', Cm_clean, time_clean),
    extrema_row('Cl_theodorsen', cl_theodorsen_cycle, time_cycle),
]
save_peak_values_table(peak_rows, 'peak_values.csv')
print("Saved → peak_values.csv")

if os.environ.get('MPLBACKEND', '').lower() != 'agg':
    plt.show()

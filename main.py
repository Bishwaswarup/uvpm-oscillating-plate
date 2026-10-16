import json
import os

import numpy as np

from config import configure_runtime_environment, get_params, get_grid, write_snapshot
from kinematics import get_kinematics, update_geometry
from forces import calculate_pressure_distribution
from simulate import simulate

PATHS = configure_runtime_environment()
FIG = PATHS['figures']
RES = PATHS['results']

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

sim = simulate(params, grid)
print("Simulation complete.")

wake_X, wake_Z, wake_Gamma = sim['wake_X'], sim['wake_Z'], sim['wake_Gamma']
geometry = sim['geometry']
Gamma_bound_final, Gamma_bound_final_prev = sim['Gamma_bound'], sim['Gamma_bound_prev']
Cl_history, Cd_history, Cm_history = sim['Cl'], sim['Cd'], sim['Cm']
alpha_history, h_history = sim['alpha_deg'], sim['h']

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

# Discard startup transient (set in config.py via cutoff_periods)
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
plt.savefig(FIG / 'cl_history.png', dpi=200)
print("Saved → figures/cl_history.png")

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
plt.savefig(FIG / 'cd_history.png', dpi=200)
print("Saved → figures/cd_history.png")

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
plt.savefig(FIG / 'cm_history.png', dpi=200)
print("Saved → figures/cm_history.png")

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
plt.savefig(FIG / 'alpha_hysteresis.png', dpi=200)
print("Saved → figures/alpha_hysteresis.png")

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
plt.savefig(FIG / 'wake_after_plate.png', dpi=200)
print("Saved → figures/wake_after_plate.png")

save_kinematics_plot(time_arr, h_arr, alpha_deg, params, FIG / 'kinematics.png')
print("Saved → figures/kinematics.png")

phase_times = [0.0, 0.25 * params['_T_osc'], 0.5 * params['_T_osc'], 0.75 * params['_T_osc'], params['_T_osc']]
phase_geometries = [update_geometry(t_phase, params, grid) for t_phase in phase_times]
save_motion_phase_schematic(phase_times, phase_geometries, params, FIG / 'motion_phase_schematic.png')
print("Saved → figures/motion_phase_schematic.png")

schematic_geometry = update_geometry(0.0, params, grid)
save_panel_schematic(params, grid, schematic_geometry, FIG / 'panel_schematic.png')
print("Saved → figures/panel_schematic.png")

h_effective, h_dot_effective, alpha_effective, _ = get_kinematics(0.0, params)
effective_geometry = update_geometry(0.0, params, grid)
save_effective_alpha_sketch(
    effective_geometry,
    h_dot_effective,
    params,
    FIG / 'effective_alpha_sketch.png',
)
print("Saved → figures/effective_alpha_sketch.png")

_, Cp_jump_final = calculate_pressure_distribution(
    Gamma_bound_final, Gamma_bound_final_prev, params, grid
)
save_cp_distribution_plot(
    grid['x_c_local'] / params['c'],
    Cp_jump_final,
    time_arr[-1],
    FIG / 'cp_distribution.png',
)
print("Saved → figures/cp_distribution.png")

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
    FIG / 'theodorsen_comparison.png',
)
print("Saved → figures/theodorsen_comparison.png")

peak_rows = [
    extrema_row('h_m', h_arr, time_arr),
    extrema_row('alpha_deg', alpha_deg, time_arr),
    extrema_row('Cl', Cl_clean, time_clean),
    extrema_row('Cd_pressure', Cd_clean, time_clean),
    extrema_row('Cm_le', Cm_clean, time_clean),
    extrema_row('Cl_theodorsen', cl_theodorsen_cycle, time_cycle),
]
save_peak_values_table(peak_rows, RES / 'peak_values.csv')
print("Saved → results/peak_values.csv")

# Amplitude and phase of the numerical lift against Theodorsen over the
# post-transient record (first harmonic at the forcing frequency).
from post_processing import harmonic_amplitude
cl_th_clean, cl_qs_clean, _ = theodorsen_lift_series(
    alpha_rad_arr[cutoff:], h_arr[cutoff:], time_clean, params['omega'], params['c'], params['U_inf']
)
amp_num = harmonic_amplitude(Cl_clean, time_clean, params['omega'])
amp_th = harmonic_amplitude(cl_th_clean, time_clean, params['omega'])
summary = {
    'preset': params['_preset'],
    'N': params['N'], 'dt': params['dt'], 'T_total': params['T_total'],
    'wake_chords': params['wake_chords'], 'rc': params['rc'],
    'reduced_frequency_k': params['_k_reduced'],
    'theodorsen_C_k': [float(c_k.real), float(c_k.imag)],
    'cl_amplitude_uvpm': float(abs(amp_num)),
    'cl_amplitude_theodorsen': float(abs(amp_th)),
    'cl_amplitude_ratio': float(abs(amp_num) / abs(amp_th)),
    'cl_phase_difference_deg': float(np.degrees(np.angle(amp_num / amp_th))),
    'cm_le_amplitude': float(abs(harmonic_amplitude(Cm_clean, time_clean, params['omega']))),
    'cd_pressure_max': float(Cd_clean.max()),
    'wake_vortices_final': int(len(wake_Gamma)),
}
with open(RES / 'summary.json', 'w') as fh:
    json.dump(summary, fh, indent=2)
print("Saved → results/summary.json")
print(f"C_L amplitude: UVPM {summary['cl_amplitude_uvpm']:.5f}, Theodorsen {summary['cl_amplitude_theodorsen']:.5f} "
      f"(ratio {summary['cl_amplitude_ratio']:.4f}, phase {summary['cl_phase_difference_deg']:+.2f} deg)")

if os.environ.get('MPLBACKEND', '').lower() != 'agg':
    plt.show()

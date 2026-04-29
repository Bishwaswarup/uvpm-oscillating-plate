import csv
import numpy as np


def theodorsen_function(k):
    if abs(k) < 1e-12:
        return 1.0 + 0.0j
    return 1.0 - 0.165 / (1.0 - 0.0455j / k) - 0.335 / (1.0 - 0.3j / k)


def harmonic_amplitude(signal, time, omega):
    return 2.0 * np.mean(signal * np.exp(-1j * omega * time))


def theodorsen_lift_series(alpha, h, time, omega, chord, u_inf):
    alpha_hat = harmonic_amplitude(alpha, time, omega)
    h_hat = harmonic_amplitude(h, time, omega)
    iomega = 1j * omega
    c_k = theodorsen_function(omega * chord / (2.0 * u_inf))

    # Upward-positive h uses the opposite sign from the common plunge-downward convention.
    alpha_eff_hat = alpha_hat - iomega * h_hat / u_inf + 0.75 * chord * iomega * alpha_hat / u_inf
    cl_circ_hat = 2.0 * np.pi * c_k * alpha_eff_hat

    cl_nc_hat = (
        np.pi * chord / (2.0 * u_inf ** 2)
        * (u_inf * iomega * alpha_hat + 0.5 * chord * (iomega ** 2) * alpha_hat - (iomega ** 2) * h_hat)
    )

    cl_total = np.real((cl_circ_hat + cl_nc_hat) * np.exp(1j * omega * time))
    cl_quasi_steady = np.real(2.0 * np.pi * alpha_eff_hat * np.exp(1j * omega * time))
    return cl_total, cl_quasi_steady, c_k


def save_kinematics_plot(time, h, alpha_deg, params, output_path):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f'Prescribed Kinematics  [k={params["_k_reduced"]:.4f}, T={params["_T_osc"]:.2f} s]',
        fontsize=13
    )
    axs[0].plot(time, h, color='tab:blue')
    axs[0].set_ylabel('$h$ (m)')
    axs[0].grid(True)
    axs[1].plot(time, alpha_deg, color='tab:orange')
    axs[1].set_ylabel('$\\alpha$ (deg)')
    axs[1].set_xlabel('Time (s)')
    axs[1].grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_theodorsen_comparison_plot(
    time_cycle,
    cl_numerical,
    cl_quasi_steady,
    cl_theodorsen,
    c_k,
    params,
    output_path,
):
    import matplotlib.pyplot as plt

    time_local = time_cycle - time_cycle[0]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(time_local, cl_numerical, label='Numerical UVPM', linewidth=2)
    ax.plot(time_local, cl_quasi_steady, '--', label='Quasi-steady', alpha=0.8)
    ax.plot(time_local, cl_theodorsen, '-.', label='Theodorsen', alpha=0.9)
    ax.set_title(
        f'Lift Comparison Over Final Cycle  [k={params["_k_reduced"]:.4f}]',
        fontsize=13
    )
    ax.set_xlabel('Time in Final Cycle (s)')
    ax.set_ylabel('$C_L$')
    ax.grid(True)
    ax.legend()
    summary = (
        f'C(k) = {c_k.real:.4f} {c_k.imag:+.4f}i\n'
        f'|C(k)| = {abs(c_k):.4f}, phase = {np.degrees(np.angle(c_k)):.2f} deg'
    )
    ax.text(
        0.02,
        0.97,
        summary,
        transform=ax.transAxes,
        va='top',
        bbox={'facecolor': 'white', 'edgecolor': '0.8', 'alpha': 0.9},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_panel_schematic(params, grid, geometry, output_path):
    import matplotlib.pyplot as plt

    x_nodes_local = grid['x_nodes_local']
    ca = np.cos(geometry['alpha'])
    sa = np.sin(geometry['alpha'])
    x_nodes = x_nodes_local * ca
    z_nodes = geometry['h'] - x_nodes_local * sa
    stride = max(1, params['N'] // 12)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for i in range(params['N']):
        ax.plot(x_nodes[i:i + 2], z_nodes[i:i + 2], color='k', linewidth=1.2)

    ax.scatter(x_nodes, z_nodes, s=12, color='k', label='Panel nodes')
    ax.scatter(
        geometry['X_v'][::stride],
        geometry['Z_v'][::stride],
        s=28,
        color='tab:red',
        marker='o',
        label='Quarter-chord vortex points',
    )
    ax.scatter(
        geometry['X_c'][::stride],
        geometry['Z_c'][::stride],
        s=28,
        color='tab:blue',
        marker='s',
        label='Three-quarter-chord collocation points',
    )
    ax.quiver(
        geometry['X_c'][::stride],
        geometry['Z_c'][::stride],
        0.08 * params['c'] * geometry['n_X'][::stride],
        0.08 * params['c'] * geometry['n_Z'][::stride],
        angles='xy',
        scale_units='xy',
        scale=1.0,
        color='tab:green',
        width=0.003,
        label='Panel normals',
    )
    ax.text(x_nodes[0], z_nodes[0] + 0.02 * params['c'], 'LE')
    ax.text(x_nodes[-1], z_nodes[-1] + 0.02 * params['c'], 'TE', ha='right')
    ax.set_title(
        f'Panel Discretization Schematic  [N={params["N"]}, shown at t=0 s]',
        fontsize=13
    )
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.grid(True)
    ax.axis('equal')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_cp_distribution_plot(x_over_c, cp_jump, time_value, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_over_c, cp_jump, 'o-', color='tab:purple', markersize=4)
    ax.set_title(
        f'Instantaneous Pressure-Jump Distribution at t={time_value:.2f} s',
        fontsize=13
    )
    ax.set_xlabel('$x/c$')
    ax.set_ylabel('$\\Delta C_p$')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_peak_values_table(rows, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['quantity', 'min_value', 'time_at_min_s', 'max_value', 'time_at_max_s'])
        writer.writerows(rows)


def save_motion_phase_schematic(times, geometries, params, output_path):
    import matplotlib.pyplot as plt

    nrows = len(times)
    fig, axs = plt.subplots(nrows, 1, figsize=(6.0, 2.4 * nrows), sharex=True)
    if nrows == 1:
        axs = [axs]

    x_local = np.array([0.0, params['c']])
    z_all = []
    for geom in geometries:
        ca = np.cos(geom['alpha'])
        sa = np.sin(geom['alpha'])
        z_plate = geom['h'] - x_local * sa
        z_all.extend(z_plate.tolist())
        z_all.append(geom['h'])

    z_pad = 0.18 * params['c']
    z_min = min(z_all) - z_pad
    z_max = max(z_all) + z_pad

    for ax, t, geom in zip(axs, times, geometries):
        ca = np.cos(geom['alpha'])
        sa = np.sin(geom['alpha'])
        x_plate = x_local * ca
        z_plate = geom['h'] - x_local * sa
        ax.plot(x_plate, z_plate, 'k-', linewidth=4)
        ax.scatter(x_plate, z_plate, color='k', s=20)
        ax.axhline(0.0, color='0.8', linestyle='--', linewidth=1)
        ax.annotate(
            '',
            xy=(-0.08 * params['c'], geom['h'] + 0.8 * params['h_amp']),
            xytext=(-0.08 * params['c'], geom['h']),
            arrowprops={'arrowstyle': '->', 'color': 'tab:blue', 'lw': 2},
        )
        ax.text(
            0.02 * params['c'],
            z_max - 0.08 * (z_max - z_min),
            f't = {t:.1f} s\nh = {geom["h"]:.3f} m\n$\\alpha$ = {np.degrees(geom["alpha"]):.2f}$^\\circ$',
            fontsize=9,
            va='top',
        )
        ax.set_xlim(-0.05 * params['c'], 1.02 * params['c'])
        ax.set_ylim(z_min, z_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x/c')
        ax.set_xticks([0.0, 0.5 * params['c'], params['c']])
        ax.set_xticklabels(['0', '0.5', '1'])

    for ax in axs:
        ax.set_ylabel('z (m)')
    fig.suptitle('Motion Phase Schematic Over One Oscillation Cycle', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_effective_alpha_sketch(geometry, h_dot, params, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    x_plate_local = np.array([0.0, params['c']])
    ca = np.cos(geometry['alpha'])
    sa = np.sin(geometry['alpha'])
    x_plate = x_plate_local * ca
    z_plate = geometry['h'] - x_plate_local * sa
    mid_x = 0.5 * (x_plate[0] + x_plate[1])
    mid_z = 0.5 * (z_plate[0] + z_plate[1])

    ax.plot(x_plate, z_plate, 'k-', linewidth=4, label='Flat plate')
    ax.scatter(x_plate, z_plate, color='k', s=25)

    flow_scale = 0.28 * params['c']
    vel_scale = 120.0
    alpha_eff = geometry['alpha'] - h_dot / params['U_inf']

    ax.annotate(
        '',
        xy=(mid_x - flow_scale, mid_z),
        xytext=(mid_x + 0.02 * params['c'], mid_z),
        arrowprops={'arrowstyle': '->', 'lw': 2.5, 'color': 'tab:blue'},
    )
    ax.text(mid_x - 0.65 * flow_scale, mid_z + 0.03 * params['c'], '$V_\\infty$', color='tab:blue')

    ax.annotate(
        '',
        xy=(mid_x, mid_z + h_dot * vel_scale),
        xytext=(mid_x, mid_z),
        arrowprops={'arrowstyle': '->', 'lw': 2.5, 'color': 'tab:orange'},
    )
    ax.text(mid_x + 0.02 * params['c'], mid_z + 0.5 * h_dot * vel_scale, '$\\dot{h}$', color='tab:orange')

    vrel_x = -params['U_inf']
    vrel_z = -h_dot
    vrel_norm = np.hypot(vrel_x, vrel_z)
    rel_scale = flow_scale / max(vrel_norm / params['U_inf'], 1e-12)
    ax.annotate(
        '',
        xy=(mid_x + vrel_x / params['U_inf'] * rel_scale, mid_z + vrel_z / params['U_inf'] * rel_scale),
        xytext=(mid_x, mid_z),
        arrowprops={'arrowstyle': '->', 'lw': 2.5, 'color': 'tab:green'},
    )
    ax.text(
        mid_x - 0.62 * flow_scale,
        mid_z - 0.12 * params['c'],
        '$V_{rel}$',
        color='tab:green',
    )

    arc_r = 0.16 * params['c']
    theta_plate = -geometry['alpha']
    theta_flow = alpha_eff
    arc1 = np.linspace(0.0, theta_plate, 60)
    arc2 = np.linspace(0.0, theta_flow, 60)
    ax.plot(mid_x + arc_r * np.cos(arc1), mid_z + arc_r * np.sin(arc1), color='tab:red', linewidth=1.8)
    ax.plot(mid_x + 1.15 * arc_r * np.cos(arc2), mid_z + 1.15 * arc_r * np.sin(arc2), color='tab:purple', linewidth=1.8)
    ax.text(mid_x + 0.16 * params['c'], mid_z - 0.06 * params['c'], '$\\alpha$', color='tab:red')
    ax.text(mid_x + 0.12 * params['c'], mid_z + 0.08 * params['c'], '$\\alpha_{eff}$', color='tab:purple')

    ax.axhline(0.0, color='0.85', linestyle='--', linewidth=1)
    ax.set_title('Effective Angle-of-Attack Sketch', fontsize=14)
    ax.text(
        0.03,
        0.96,
        f'$\\alpha_{{eff}} \\approx \\alpha - \\dot{{h}}/V_\\infty$\\n'
        f'$\\alpha = {np.degrees(geometry["alpha"]):.2f}^\\circ$, '
        f'$\\dot{{h}} = {h_dot:.4f}$ m/s',
        transform=ax.transAxes,
        va='top',
        bbox={'facecolor': 'white', 'edgecolor': '0.8', 'alpha': 0.9},
    )
    ax.set_xlim(-0.4 * params['c'], 1.05 * params['c'])
    ax.set_ylim(geometry['h'] - 0.35 * params['c'], geometry['h'] + 0.35 * params['c'])
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

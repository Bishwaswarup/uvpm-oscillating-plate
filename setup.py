"""
setup.py — Simulation Parameter Configuration
AE 221 Aerodynamics | Term Paper | Question 3
Unsteady Vortex Panel Method for Oscillating Flat Plate

HOW TO USE:
  Edit the values in SimulationConfig below, then run:
    python setup.py          → validates and previews your settings
    python main.py           → runs the actual simulation

COMPUTATIONAL COST GUIDE:
  Total operations ≈ N_steps × N² × M_wake
  where M_wake = min(N_steps, wake_limit / (U_inf × dt))

  Preset modes (see PRESET at bottom):
    'fast'      → ~5  seconds  | good for debugging
    'balanced'  → ~30 seconds  | recommended for term paper
    'accurate'  → ~3  minutes  | high resolution results
    'research'  → ~15 minutes  | maximum fidelity
"""

import json
import os
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────
#  EDIT YOUR PARAMETERS HERE
# ─────────────────────────────────────────────

PRESET = 'research'   # 'fast' | 'balanced' | 'accurate' | 'research' | 'custom'

# Only used when PRESET = 'custom'
CUSTOM = {
    # ── Physics (do not change for Q3) ──────────────────────────────
    'U_inf'  : 40.0,          # Free stream velocity (m/s)
    'c'      : 1.0,           # Chord length (m)
    'rho'    : 1.225,         # Air density (kg/m³)
    'omega'  : 0.1 * np.pi,  # Oscillation frequency (rad/s) — from problem statement
    'h_amp'  : 0.01,          # Heave amplitude (m)
    'alpha_amp': 0.01,        # Pitch amplitude (rad)

    # ── Numerical resolution ─────────────────────────────────────────
    'grid_spacing': 'cosine',  # 'uniform' | 'cosine'

    # N: Number of chordwise panels
    #   ↑ N → better pressure/circulation distribution, slower build_aic (N² cost)
    #   Sweet spot: 20–50. Beyond 80 gives <1% improvement for flat plate.
    'N'      : 50,

    # dt: Time step (seconds)
    #   ↓ dt → more accurate wake convection, more steps needed
    #   Stability limit: dt < c / (2 × U_inf) = 0.0125 s  ← never go below this
    #   For k=0.004 (quasi-steady), dt=0.05 is already over-resolved
    'dt'     : 0.0125,

    # T_total: Total simulation time (seconds)
    #   Should be at least 3× the oscillation period for converged hysteresis
    #   Period T = 2π/ω = 2π/(0.1π) = 20 s  →  minimum T_total = 60 s
    'T_total': 60.0,

    # ── Wake truncation ──────────────────────────────────────────────
    # wake_chords: How many chord lengths of wake to retain
    #   ↑ wake_chords → more accurate far-wake influence, larger M×M matrix
    #   For k=0.004: beyond 25c influence is <0.1% of total Cl
    #   For k>0.1  : need at least 50c for accurate results
    'wake_chords': 50,

    # ── Numerics ─────────────────────────────────────────────────────
    # rc: Vortex core radius (fraction of chord)
    #   Too small → wake explosions when vortices get close
    #   Too large → artificially damps wake roll-up
    'rc_fraction': 0.05,      # rc = rc_fraction × c

    # epsilon: New wake vortex placement (fraction of U_inf × dt downstream of TE)
    #   Katz & Plotkin §13.8 recommend 0.5
    'epsilon': 0.5,

    # cutoff_periods: How many oscillation periods to discard as startup transient
    'cutoff_periods': 1,
}

# ─────────────────────────────────────────────
#  PRESETS  (override CUSTOM when PRESET != 'custom')
# ─────────────────────────────────────────────

PRESETS = {
    'fast': {
        'N': 20, 'dt': 0.1, 'T_total': 60.0,
        'wake_chords': 25, 'rc_fraction': 0.05,
        'epsilon': 0.5, 'cutoff_periods': 1,
        '_note': 'Good for debugging. Wake has only ~10 vortices. Cl error ~5%.'
    },
    'balanced': {
        'N': 50, 'dt': 0.0125, 'T_total': 60.0,
        'wake_chords': 50, 'rc_fraction': 0.05,
        'epsilon': 0.5, 'cutoff_periods': 1,
        '_note': 'Recommended for term paper. ~25 wake vortices. Cl error <1%.'
    },
    'accurate': {
        'N': 80, 'dt': 0.0125, 'T_total': 80.0,
        'wake_chords': 80, 'rc_fraction': 0.03,
        'epsilon': 0.5, 'cutoff_periods': 2,
        '_note': 'High fidelity. ~40 wake vortices. Cl error <0.1%.'
    },
    'research': {
        'N': 100, 'dt': 0.01, 'T_total': 100.0,
        'wake_chords': 100, 'rc_fraction': 0.02,
        'epsilon': 0.5, 'cutoff_periods': 2,
        '_note': 'Maximum fidelity. ~100 wake vortices. For convergence studies.'
    },
}

# Physics constants (same for all presets)
PHYSICS = {
    'U_inf'      : 40.0,
    'c'          : 1.0,
    'rho'        : 1.225,
    'omega'      : 0.1 * np.pi,
    'h_amp'      : 0.01,
    'alpha_amp'  : 0.01,
    'alpha_phase': 0.5 * np.pi,
}

# ─────────────────────────────────────────────
#  BUILD FINAL PARAMS  (called by main.py)
# ─────────────────────────────────────────────

def get_selected_config():
    if PRESET == 'custom':
        return CUSTOM.copy()
    if PRESET not in PRESETS:
        valid = "', '".join(sorted(PRESETS))
        raise ValueError(f"Unknown PRESET '{PRESET}'. Choose 'custom' or one of '{valid}'.")
    return {**PHYSICS, **PRESETS[PRESET]}


def build_params(cfg):
    T_osc = 2 * np.pi / cfg['omega']
    N_steps = int(cfg['T_total'] / cfg['dt'])
    k_red = (cfg['omega'] * cfg['c']) / (2 * cfg['U_inf'])
    rc = cfg['rc_fraction'] * cfg['c']

    vortex_spacing = cfg['U_inf'] * cfg['dt']
    max_wake_len = cfg['wake_chords'] * cfg['c']
    M_max = int(max_wake_len / vortex_spacing)
    cutoff = int(cfg['cutoff_periods'] * T_osc / cfg['dt'])

    return {
        'U_inf'         : cfg['U_inf'],
        'c'             : cfg['c'],
        'rho'           : cfg['rho'],
        'omega'         : cfg['omega'],
        'h_amp'         : cfg['h_amp'],
        'alpha_amp'     : cfg['alpha_amp'],
        'alpha_phase'   : cfg.get('alpha_phase', 0.5 * np.pi),
        'grid_spacing'  : cfg.get('grid_spacing', 'cosine'),
        'N'             : cfg['N'],
        'dt'            : cfg['dt'],
        'T_total'       : cfg['T_total'],
        'N_steps'       : N_steps,
        'rc'            : rc,
        'epsilon'       : cfg['epsilon'],
        'wake_chords'   : cfg['wake_chords'],
        'cutoff'        : cutoff,
        'cutoff_periods': cfg['cutoff_periods'],
        '_k_reduced'    : k_red,
        '_T_osc'        : T_osc,
        '_M_max_wake'   : M_max,
        '_vortex_spacing': vortex_spacing,
        '_preset'       : PRESET,
    }


def get_params():
    return build_params(get_selected_config())


def get_grid(params):
    """Returns the chordwise discretization grid."""
    if params['grid_spacing'] == 'cosine':
        beta = np.linspace(0.0, np.pi, params['N'] + 1)
        x_node_local = 0.5 * params['c'] * (1.0 - np.cos(beta))
    else:
        x_node_local = np.linspace(0.0, params['c'], params['N'] + 1)

    panel_dx = np.diff(x_node_local)
    grid = {
        'dx'           : panel_dx,
        'x_nodes_local': x_node_local,
        'x_v_local'    : x_node_local[:-1] + 0.25 * panel_dx,
        'x_c_local'    : x_node_local[:-1] + 0.75 * panel_dx,
    }
    return grid


def get_runtime_paths(base_dir=None):
    root = Path(base_dir or Path(__file__).resolve().parent)
    cache_root = root / '.cache'
    return {
        'root': root,
        'cache_root': cache_root,
        'mpl_cache': cache_root / 'matplotlib',
        'fontconfig_cache': cache_root / 'fontconfig',
        'snapshot': root / 'simulation_params_snapshot.json',
    }


def configure_runtime_environment(base_dir=None):
    paths = get_runtime_paths(base_dir)
    paths['mpl_cache'].mkdir(parents=True, exist_ok=True)
    paths['fontconfig_cache'].mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(paths['mpl_cache']))
    os.environ.setdefault('XDG_CACHE_HOME', str(paths['cache_root']))
    os.environ.setdefault('FONTCONFIG_PATH', '/opt/homebrew/etc/fonts')
    if not os.environ.get('DISPLAY'):
        os.environ.setdefault('MPLBACKEND', 'Agg')
    return paths


def write_snapshot(params, base_dir=None):
    paths = get_runtime_paths(base_dir)
    snapshot = {k: v for k, v in params.items() if not k.startswith('_')}
    snapshot['preset'] = PRESET
    with paths['snapshot'].open('w') as f:
        json.dump(snapshot, f, indent=2)
    return paths['snapshot']


# ─────────────────────────────────────────────
#  VALIDATION & PREVIEW  (run setup.py directly)
# ─────────────────────────────────────────────

def estimate_runtime(params):
    """Rough runtime estimate based on dominant cost (wake-on-wake)."""
    N       = params['N']
    N_steps = params['N_steps']
    M       = params['_M_max_wake']

    # Cost units: number of floating point operations (rough)
    aic_cost       = N_steps * N**2           # AIC rebuild each step
    wake_col_cost  = N_steps * N * M          # wake on collocation
    wake_wake_cost = N_steps * M**2           # wake on wake (dominant)
    total_ops      = aic_cost + wake_col_cost + wake_wake_cost

    # Calibrated against MacBook Air M2: ~2e8 ops/sec for this numpy code
    ops_per_sec = 2e8
    seconds     = total_ops / ops_per_sec

    if seconds < 10:
        return f"{seconds:.1f} seconds"
    elif seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"


def validate(params):
    """Check for common parameter mistakes."""
    warnings = []
    errors   = []

    dt_stability = params['c'] / (2 * params['U_inf'])
    if params['dt'] > dt_stability:
        errors.append(
            f"dt={params['dt']} exceeds stability limit {dt_stability:.4f}s. "
            f"Reduce dt or increase U_inf."
        )

    T_osc = params['_T_osc']
    if params['T_total'] < 2 * T_osc:
        warnings.append(
            f"T_total={params['T_total']}s is less than 2 oscillation periods "
            f"({2*T_osc:.1f}s). Hysteresis loop may not be converged."
        )

    if params['N'] < 10:
        warnings.append(f"N={params['N']} is very coarse. Use N≥20 for reasonable results.")

    if params['_M_max_wake'] < 5:
        warnings.append(
            f"Only {params['_M_max_wake']} wake vortices retained. "
            f"Increase wake_chords for better accuracy."
        )

    if params['rc'] > 0.1 * params['c']:
        warnings.append(
            f"rc={params['rc']:.3f}m ({params['rc']/params['c']*100:.0f}% chord) is large. "
            f"May damp wake interactions. Consider rc_fraction ≤ 0.05."
        )

    return warnings, errors


def preview():
    params = get_params()
    grid   = get_grid(params)
    warnings, errors = validate(params)

    print("\n" + "═"*55)
    print("  AE 221 — Unsteady Panel Method — Setup Preview")
    print("═"*55)

    preset_label = PRESET.upper()
    if PRESET != 'custom' and '_note' in PRESETS[PRESET]:
        print(f"\n  Preset : {preset_label}")
        print(f"  Note   : {PRESETS[PRESET]['_note']}")

    print(f"\n  ── Physics ──────────────────────────────────────")
    print(f"  U_inf          = {params['U_inf']} m/s")
    print(f"  chord          = {params['c']} m")
    print(f"  rho            = {params['rho']} kg/m³")
    print(f"  omega          = {params['omega']:.4f} rad/s")
    print(f"  Reduced freq k = {params['_k_reduced']:.4f}  (k<0.05 → quasi-steady)")
    print(f"  Osc. period T  = {params['_T_osc']:.2f} s")

    print(f"\n  ── Numerical Resolution ─────────────────────────")
    print(f"  N panels       = {params['N']}")
    print(f"  Grid spacing   = {params['grid_spacing']}")
    print(f"  dt             = {params['dt']} s")
    print(f"  N_steps        = {params['N_steps']}")
    print(f"  T_total        = {params['T_total']} s  ({params['T_total']/params['_T_osc']:.1f} periods)")
    print(f"  rc             = {params['rc']:.4f} m  ({params['rc']/params['c']*100:.1f}% chord)")
    print(f"  epsilon        = {params['epsilon']}")

    print(f"\n  ── Wake ─────────────────────────────────────────")
    print(f"  Retained wake  = {params['wake_chords']} chords  ({params['wake_chords']*params['c']:.1f} m)")
    print(f"  Vortex spacing = {params['_vortex_spacing']:.3f} m  (U_inf × dt)")
    print(f"  Max M_wake     = {params['_M_max_wake']} vortices at steady state")
    print(f"  Cutoff         = {params['cutoff']} steps  ({params['cutoff_periods']} period(s) discarded)")

    print(f"\n  ── Cost Estimate ────────────────────────────────")
    print(f"  Est. runtime   ≈ {estimate_runtime(params)}")
    print(f"  Dominant cost  : wake-on-wake  O(N_steps × M²) = "
          f"O({params['N_steps']} × {params['_M_max_wake']}²)")

    if warnings:
        print(f"\n  ── Warnings ─────────────────────────────────────")
        for w in warnings:
            print(f"  ⚠  {w}")

    if errors:
        print(f"\n  ── Errors (fix before running) ──────────────────")
        for e in errors:
            print(f"  ✗  {e}")
    else:
        print(f"\n  ✓  Parameters valid. Run:  python main.py")

    print("═"*55 + "\n")

    snapshot_path = write_snapshot(params)
    print(f"  Params saved → {snapshot_path.name}\n")


if __name__ == '__main__':
    preview()

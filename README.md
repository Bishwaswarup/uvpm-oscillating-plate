# UVPM Oscillating Plate

This repository contains a modular Python implementation of the Unsteady Vortex Panel Method (UVPM), developed to analyze the unsteady aerodynamic characteristics of a flat plate executing small-amplitude heave and pitch oscillations.

Built based on the theoretical frameworks in *Low-Speed Aerodynamics* by Katz & Plotkin, the solver enforces the zero normal flow boundary condition and utilizes the unsteady Bernoulli equation to compute time-varying aerodynamic forces (`C_L`, `C_D`, `C_M,LE`).

## Key Features

- `Kinematic Modeling`: Accurately simulates coupled heave and pitch motions with customizable phase shifts.
- `Wake Management`: Implements explicit forward-Euler wake convection and dynamic vortex shedding governed by Kelvin's circulation theorem.
- `Force Calculation`: Calculates unsteady lift, pitching moment, and induced pressure drag, highlighting phenomena like the leading-edge suction peak and the Kutta condition.

## Repository Contents

- `main.py`: Runs the full unsteady flat-plate simulation and generates output figures/data.
- `setup.py`: Centralized simulation configuration, presets, derived parameters, and runtime setup.
- `kinematics.py`: Heave-pitch motion definition and plate geometry updates.
- `solver.py`: Boundary-condition assembly and bound-circulation solution.
- `forces.py`: Unsteady pressure-jump, lift, drag, and leading-edge moment calculations.
- `wake_management.py`: Wake convection and newly shed vortex placement.
- `vortex_math.py`: Biot-Savart influence calculations and AIC assembly.
- `post_processing.py`: Plotting helpers, Theodorsen comparison, and reporting utilities.

## Generated Outputs

Typical runs produce:

- Time-history plots for `C_L`, `C_D`, and `C_M`
- Hysteresis plots versus angle of attack
- Wake visualization
- Kinematics plot
- Theodorsen comparison
- Panel discretization schematic
- Effective-angle and motion-phase explanatory visuals
- Instantaneous pressure-jump distribution
- Peak-value summary table

## Running the Solver

Preview the selected setup:

```bash
python setup.py
```

Run the simulation:

```bash
python main.py
```

## Reference

- Joseph Katz and Allen Plotkin, *Low-Speed Aerodynamics*

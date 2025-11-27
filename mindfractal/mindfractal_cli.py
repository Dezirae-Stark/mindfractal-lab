#!/usr/bin/env python3
"""
MindFractal Lab - Command Line Interface

Provides easy access to simulation, visualization, and analysis functions
from the command line.

Usage:
    python -m mindfractal.mindfractal_cli simulate --x0 0.5 0.5 --steps 1000
    python -m mindfractal.mindfractal_cli visualize --output orbit.png
    python -m mindfractal.mindfractal_cli fractal --resolution 500 --output fractal.png
    python -m mindfractal.mindfractal_cli analyze --mode lyapunov
    python -m mindfractal.mindfractal_cli td slice --steps 20 --output timeline.png
    python -m mindfractal.mindfractal_cli td visualize --resolution 100 --output landscape.png
"""

import argparse
import sys

import numpy as np

from .fractal_map import generate_fractal_map
from .model import FractalDynamicsModel
from .simulate import compute_attractor_type, find_fixed_points, simulate_orbit
from .visualize import (
    plot_basin_of_attraction,
    plot_fractal_map,
    plot_lyapunov_spectrum,
    plot_orbit,
)

# Optional: Tenth Dimension Possibility Module
try:
    from extensions.tenth_dimension_possibility import possibility_cli as td_cli

    TD_AVAILABLE = True
except ImportError:
    TD_AVAILABLE = False


def cmd_simulate(args):
    """Simulate an orbit."""
    print("=== MindFractal Lab: Orbit Simulation ===")

    # Parse initial condition
    if args.x0:
        x0 = np.array(args.x0)
    else:
        x0 = np.array([0.5, 0.5])

    print(f"Initial condition: x0 = {x0}")
    print(f"Simulation steps: {args.steps}")

    # Create model
    model = FractalDynamicsModel()
    print(f"Model: {model}")

    # Simulate
    trajectory = simulate_orbit(model, x0, n_steps=args.steps)

    print(f"\nTrajectory shape: {trajectory.shape}")
    print(f"Final state: {trajectory[-1]}")
    print(f"Final norm: {np.linalg.norm(trajectory[-1]):.6f}")

    # Determine attractor type
    atype = compute_attractor_type(model, x0, n_steps=args.steps)
    print(f"Attractor type: {atype}")

    # Lyapunov exponent
    lyap = model.lyapunov_exponent_estimate(x0, n_steps=min(args.steps, 2000))
    print(f"Lyapunov exponent: {lyap:.6f}")

    if args.output:
        np.savetxt(args.output, trajectory, fmt="%.8f", header="x1 x2")
        print(f"\nTrajectory saved to {args.output}")


def cmd_visualize(args):
    """Visualize an orbit."""
    print("=== MindFractal Lab: Visualization ===")

    x0 = np.array(args.x0) if args.x0 else np.array([0.5, 0.5])
    print(f"Initial condition: x0 = {x0}")

    model = FractalDynamicsModel()

    if args.mode == "orbit":
        fig = plot_orbit(model, x0, n_steps=args.steps, save_path=args.output)
        print(f"Orbit plot saved to {args.output}")

    elif args.mode == "basin":
        fig = plot_basin_of_attraction(model, resolution=args.resolution, save_path=args.output)
        print(f"Basin of attraction plot saved to {args.output}")

    elif args.mode == "lyapunov":
        fig = plot_lyapunov_spectrum(model, x0, n_steps=args.steps, save_path=args.output)
        print(f"Lyapunov spectrum plot saved to {args.output}")


def cmd_fractal(args):
    """Generate fractal parameter-space map."""
    print("=== MindFractal Lab: Fractal Map Generation ===")

    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"c1 range: {args.c1_range}")
    print(f"c2 range: {args.c2_range}")
    print(f"Criterion: {args.criterion}")

    fractal_data = generate_fractal_map(
        c1_range=tuple(args.c1_range),
        c2_range=tuple(args.c2_range),
        resolution=args.resolution,
        max_steps=args.max_steps,
        criterion=args.criterion,
    )

    if args.output:
        fig = plot_fractal_map(
            fractal_data,
            c1_range=tuple(args.c1_range),
            c2_range=tuple(args.c2_range),
            save_path=args.output,
        )
        print(f"Fractal map saved to {args.output}")

    if args.save_data:
        np.save(args.save_data, fractal_data)
        print(f"Fractal data saved to {args.save_data}")


def cmd_analyze(args):
    """Analyze the dynamical system."""
    print("=== MindFractal Lab: Analysis ===")

    model = FractalDynamicsModel()

    if args.mode == "fixed_points":
        print("Finding fixed points...")
        fixed_points = find_fixed_points(model)

        print(f"\nFound {len(fixed_points)} fixed points:")
        for i, (fp, stable) in enumerate(fixed_points):
            stability_str = "STABLE" if stable else "UNSTABLE"
            print(f"  FP {i+1}: {fp} [{stability_str}]")

            # Compute Jacobian eigenvalues
            J = model.jacobian(fp)
            eigvals = np.linalg.eigvals(J)
            print(f"         Eigenvalues: {eigvals}")

    elif args.mode == "lyapunov":
        x0 = np.array(args.x0) if args.x0 else np.array([0.5, 0.5])
        print(f"Computing Lyapunov exponent from x0 = {x0}")

        lyap = model.lyapunov_exponent_estimate(x0, n_steps=5000, transient=1000)
        print(f"\nLargest Lyapunov exponent: {lyap:.6f}")

        if lyap > 0.01:
            print("Dynamics: CHAOTIC (positive Lyapunov exponent)")
        elif lyap < -0.01:
            print("Dynamics: CONVERGENT (negative Lyapunov exponent)")
        else:
            print("Dynamics: PERIODIC or MARGINALLY STABLE")

    elif args.mode == "attractor":
        x0 = np.array(args.x0) if args.x0 else np.array([0.5, 0.5])
        print(f"Classifying attractor from x0 = {x0}")

        atype = compute_attractor_type(model, x0, n_steps=args.steps)
        print(f"\nAttractor type: {atype.upper()}")


def cmd_tenth_dimension(args):
    """Tenth Dimension: Possibility Manifold commands."""
    if not TD_AVAILABLE:
        print("ERROR: Tenth Dimension module not available.")
        print("Install with: pip install -e .[tenth_dimension]")
        print("Or ensure extensions/tenth_dimension_possibility/ exists.")
        sys.exit(1)

    print("=== MindFractal Lab: Tenth Dimension - Possibility Manifold ===")

    # Delegate to the tenth dimension CLI
    if args.td_command == "slice":
        td_cli.td_slice(args)
    elif args.td_command == "visualize":
        td_cli.td_visualize(args)
    elif args.td_command == "random-orbit":
        td_cli.td_random_orbit(args)
    elif args.td_command == "boundary-map":
        td_cli.td_boundary_map(args)
    else:
        print(f"Unknown tenth dimension command: {args.td_command}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="MindFractal Lab - Fractal Dynamical Consciousness Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Simulate an orbit")
    sim_parser.add_argument("--x0", nargs=2, type=float, help="Initial condition (x1 x2)")
    sim_parser.add_argument("--steps", type=int, default=1000, help="Number of steps")
    sim_parser.add_argument("--output", type=str, help="Save trajectory to file")

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize dynamics")
    viz_parser.add_argument(
        "--mode",
        choices=["orbit", "basin", "lyapunov"],
        default="orbit",
        help="Visualization mode",
    )
    viz_parser.add_argument("--x0", nargs=2, type=float, help="Initial condition")
    viz_parser.add_argument("--steps", type=int, default=1000, help="Number of steps")
    viz_parser.add_argument("--resolution", type=int, default=200, help="Grid resolution")
    viz_parser.add_argument("--output", type=str, required=True, help="Output image file")

    # Fractal command
    frac_parser = subparsers.add_parser("fractal", help="Generate fractal map")
    frac_parser.add_argument("--resolution", type=int, default=500, help="Grid resolution")
    frac_parser.add_argument(
        "--c1-range",
        nargs=2,
        type=float,
        default=[-1.0, 1.0],
        help="c1 parameter range",
    )
    frac_parser.add_argument(
        "--c2-range",
        nargs=2,
        type=float,
        default=[-1.0, 1.0],
        help="c2 parameter range",
    )
    frac_parser.add_argument("--max-steps", type=int, default=500, help="Max simulation steps")
    frac_parser.add_argument(
        "--criterion",
        choices=["divergence_time", "final_norm", "lyapunov", "attractor_type"],
        default="divergence_time",
        help="Fractal map criterion",
    )
    frac_parser.add_argument("--output", type=str, help="Output image file")
    frac_parser.add_argument("--save-data", type=str, help="Save raw data (NumPy .npy format)")

    # Analyze command
    ana_parser = subparsers.add_parser("analyze", help="Analyze dynamics")
    ana_parser.add_argument(
        "--mode",
        choices=["fixed_points", "lyapunov", "attractor"],
        required=True,
        help="Analysis mode",
    )
    ana_parser.add_argument("--x0", nargs=2, type=float, help="Initial condition")
    ana_parser.add_argument("--steps", type=int, default=5000, help="Number of steps")

    # Tenth Dimension command (if available)
    if TD_AVAILABLE:
        td_parser = subparsers.add_parser(
            "tenth-dimension",
            aliases=["td", "10d"],
            help="Explore Possibility Manifold (10th dimension)",
        )
        td_subparsers = td_parser.add_subparsers(
            dest="td_command", help="Tenth dimension sub-command"
        )

        # td slice
        td_slice_parser = td_subparsers.add_parser("slice", help="Create timeline slice")
        td_slice_parser.add_argument("--dim", type=int, default=2, help="Dimension (2 or 3)")
        td_slice_parser.add_argument(
            "--steps", type=int, default=20, help="Number of timeline steps"
        )
        td_slice_parser.add_argument("--output", "-o", help="Output filename")
        td_slice_parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
        td_slice_parser.add_argument("--no-show", action="store_true", help="Don't display plot")

        # td visualize
        td_viz_parser = td_subparsers.add_parser("visualize", help="Visualize stability landscape")
        td_viz_parser.add_argument("--dim", type=int, default=2, help="Dimension")
        td_viz_parser.add_argument("--resolution", type=int, default=50, help="Grid resolution")
        td_viz_parser.add_argument("--min-param", type=float, default=-2.0, help="Min parameter")
        td_viz_parser.add_argument("--max-param", type=float, default=2.0, help="Max parameter")
        td_viz_parser.add_argument("--output", "-o", help="Output filename")
        td_viz_parser.add_argument("--no-show", action="store_true", help="Don't display plot")

        # td random-orbit
        td_orbit_parser = td_subparsers.add_parser("random-orbit", help="Generate random orbit")
        td_orbit_parser.add_argument("--dim", type=int, default=2, help="Dimension")
        td_orbit_parser.add_argument("--steps", type=int, default=500, help="Number of orbit steps")
        td_orbit_parser.add_argument("--output", "-o", help="Output filename")
        td_orbit_parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
        td_orbit_parser.add_argument("--no-show", action="store_true", help="Don't display plot")

        # td boundary-map
        td_bound_parser = td_subparsers.add_parser("boundary-map", help="Map stability boundaries")
        td_bound_parser.add_argument("--dim", type=int, default=2, help="Dimension")
        td_bound_parser.add_argument("--resolution", type=int, default=100, help="Grid resolution")
        td_bound_parser.add_argument("--min-param", type=float, default=-2.0, help="Min parameter")
        td_bound_parser.add_argument("--max-param", type=float, default=2.0, help="Max parameter")
        td_bound_parser.add_argument("--output", "-o", help="Output filename")
        td_bound_parser.add_argument("--no-show", action="store_true", help="Don't display plot")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate command
    if args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "fractal":
        cmd_fractal(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command in ["tenth-dimension", "td", "10d"]:
        cmd_tenth_dimension(args)


if __name__ == "__main__":
    main()

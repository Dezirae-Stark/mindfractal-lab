"""
Command-Line Interface for Tenth Dimension Possibility Module

Provides CLI commands for exploring the Possibility Manifold.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from .possibility_manifold import PossibilityManifold, UpdateRuleFamily
from .possibility_viewer import PossibilityVisualizer
from .possibility_slicer import TimelineSlicer


def td_slice(args):
    """Execute timeline slice command"""
    print(f"Creating timeline slice with {args.steps} steps...")

    manifold = PossibilityManifold(dim=args.dim)
    slicer = TimelineSlicer(manifold)

    # Sample start and end points
    start = manifold.sample_point(rule_family=UpdateRuleFamily.TANH_2D)
    end = manifold.sample_point(rule_family=UpdateRuleFamily.TANH_2D)

    # Create timeline branch
    branch = slicer.slice_parameter_line(start, end, n_steps=args.steps)

    print(f"Timeline branch created with {len(branch.points)} points")

    # Visualize if requested
    if not args.no_plot:
        visualizer = PossibilityVisualizer(manifold)
        fig = visualizer.plot_timeline_branch(branch)
        plt.savefig(args.output or "timeline_slice.png", dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {args.output or 'timeline_slice.png'}")
        if not args.no_show:
            plt.show()

def td_visualize(args):
    """Execute manifold visualization command"""
    print(f"Visualizing {args.dim}D Possibility Manifold...")

    manifold = PossibilityManifold(dim=args.dim)
    visualizer = PossibilityVisualizer(manifold)

    # Create stability landscape
    fig = visualizer.plot_stability_landscape(
        param_range=(args.min_param, args.max_param),
        resolution=args.resolution
    )

    plt.savefig(args.output or "stability_landscape.png", dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {args.output or 'stability_landscape.png'}")

    if not args.no_show:
        plt.show()

def td_random_orbit(args):
    """Execute random orbit generation"""
    print(f"Generating random orbit with {args.steps} steps...")

    manifold = PossibilityManifold(dim=args.dim)
    point = manifold.sample_point(rule_family=UpdateRuleFamily.TANH_2D)

    orbit = manifold.compute_orbit(point, steps=args.steps)
    region = manifold.classify_stability(orbit)

    print(f"Stability: {region.value}")
    print(f"Orbit shape: {orbit.shape}")

    # Plot if 2D
    if args.dim == 2 and not args.no_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(orbit[:, 0].real, orbit[:, 0].imag, 'b-', alpha=0.6, linewidth=0.5)
        ax.plot(orbit[0, 0].real, orbit[0, 0].imag, 'go', markersize=10, label='Start')
        if not np.isnan(orbit[-1]).any():
            ax.plot(orbit[-1, 0].real, orbit[-1, 0].imag, 'ro', markersize=10, label='End')
        ax.set_xlabel('Re(z₀)')
        ax.set_ylabel('Im(z₀)')
        ax.set_title(f'Random Orbit - Stability: {region.value}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.savefig(args.output or "random_orbit.png", dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {args.output or 'random_orbit.png'}")

        if not args.no_show:
            plt.show()

def td_boundary_map(args):
    """Execute boundary mapping"""
    print(f"Mapping stability boundaries...")

    manifold = PossibilityManifold(dim=args.dim)
    visualizer = PossibilityVisualizer(manifold)

    # Create high-resolution landscape focusing on boundaries
    fig = visualizer.plot_stability_landscape(
        param_range=(args.min_param, args.max_param),
        resolution=args.resolution
    )

    plt.savefig(args.output or "boundary_map.png", dpi=200, bbox_inches='tight')
    print(f"Saved boundary map to {args.output or 'boundary_map.png'}")

    if not args.no_show:
        plt.show()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Tenth Dimension Possibility Module CLI"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # td-slice command
    slice_parser = subparsers.add_parser('slice', help='Create timeline slice')
    slice_parser.add_argument('--dim', type=int, default=2, help='Dimension')
    slice_parser.add_argument('--steps', type=int, default=20, help='Number of steps')
    slice_parser.add_argument('--output', '-o', help='Output filename')
    slice_parser.add_argument('--no-plot', action='store_true', help='Skip visualization')
    slice_parser.add_argument('--no-show', action='store_true', help='Don\'t display plot')

    # td-visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize manifold')
    viz_parser.add_argument('--dim', type=int, default=2, help='Dimension')
    viz_parser.add_argument('--resolution', type=int, default=50, help='Grid resolution')
    viz_parser.add_argument('--min-param', type=float, default=-2.0, help='Min parameter')
    viz_parser.add_argument('--max-param', type=float, default=2.0, help='Max parameter')
    viz_parser.add_argument('--output', '-o', help='Output filename')
    viz_parser.add_argument('--no-show', action='store_true', help='Don\'t display plot')

    # td-random-orbit command
    orbit_parser = subparsers.add_parser('random-orbit', help='Generate random orbit')
    orbit_parser.add_argument('--dim', type=int, default=2, help='Dimension')
    orbit_parser.add_argument('--steps', type=int, default=500, help='Number of steps')
    orbit_parser.add_argument('--output', '-o', help='Output filename')
    orbit_parser.add_argument('--no-plot', action='store_true', help='Skip visualization')
    orbit_parser.add_argument('--no-show', action='store_true', help='Don\'t display plot')

    # td-boundary-map command
    boundary_parser = subparsers.add_parser('boundary-map', help='Map stability boundaries')
    boundary_parser.add_argument('--dim', type=int, default=2, help='Dimension')
    boundary_parser.add_argument('--resolution', type=int, default=100, help='Grid resolution')
    boundary_parser.add_argument('--min-param', type=float, default=-2.0, help='Min parameter')
    boundary_parser.add_argument('--max-param', type=float, default=2.0, help='Max parameter')
    boundary_parser.add_argument('--output', '-o', help='Output filename')
    boundary_parser.add_argument('--no-show', action='store_true', help='Don\'t display plot')

    args = parser.parse_args()

    if args.command == 'slice':
        td_slice(args)
    elif args.command == 'visualize':
        td_visualize(args)
    elif args.command == 'random-orbit':
        td_random_orbit(args)
    elif args.command == 'boundary-map':
        td_boundary_map(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

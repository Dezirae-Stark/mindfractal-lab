#!/usr/bin/env python3
"""
Generate Textbook-Style LaTeX Equation Images

Creates high-quality rendered equation images for README documentation.
Uses matplotlib's mathtext (no LaTeX installation required).

Output: docs/images/equations/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Configure matplotlib for high-quality math rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Times']
rcParams['mathtext.fontset'] = 'cm'  # Computer Modern (LaTeX default)
rcParams['font.size'] = 18

# Output directory
OUTPUT_DIR = 'docs/images/equations'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def render_equation(latex_str, filename, fontsize=22, figsize=(10, 1.5),
                    bg_color='white', text_color='black', dpi=150,
                    box=False, box_color='#e8e8e8'):
    """Render a LaTeX equation to a PNG image."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Remove axes
    ax.axis('off')

    # Render equation centered
    text_obj = ax.text(0.5, 0.5, latex_str, fontsize=fontsize,
                       ha='center', va='center', color=text_color,
                       transform=ax.transAxes)

    # Add box around equation if requested
    if box:
        text_obj.set_bbox(dict(boxstyle='round,pad=0.4',
                               facecolor=box_color,
                               edgecolor='#333333',
                               linewidth=2))

    # Save with tight bounding box
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor=bg_color, edgecolor='none',
                pad_inches=0.15)
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


def main():
    print("Generating textbook-style equation images...")
    print()

    # 1. Main State Equation (boxed, prominent)
    print("1. Main state equation...")
    render_equation(
        r'$\mathbf{x}_{n+1} = A\mathbf{x}_n + B\tanh(W\mathbf{x}_n) + \mathbf{c}$',
        'state_equation.png',
        fontsize=28,
        figsize=(12, 2),
        box=True,
        box_color='#f5f5dc'
    )

    # 2. State vector definition
    print("2. State vector definition...")
    render_equation(
        r'$\mathbf{x} \in \mathbb{R}^d, \quad d \in \{2, 3\}$',
        'state_vector.png',
        fontsize=22,
        figsize=(8, 1.5)
    )

    # 3. Matrix definitions
    print("3. Matrix definitions...")
    render_equation(
        r'$A, B, W \in \mathbb{R}^{d \times d}, \quad \mathbf{c} \in \mathbb{R}^d$',
        'matrix_domains.png',
        fontsize=22,
        figsize=(10, 1.3)
    )

    # 4. Jacobian
    print("4. Jacobian matrix...")
    render_equation(
        r'$J(\mathbf{x}) = A + B \cdot \mathrm{diag}\left(\mathrm{sech}^2(W\mathbf{x})\right) \cdot W$',
        'jacobian.png',
        fontsize=22,
        figsize=(12, 1.5),
        box=True,
        box_color='#e8f4e8'
    )

    # 5. Sech squared identity
    print("5. Sech identity...")
    render_equation(
        r'$\mathrm{sech}^2(z) = 1 - \tanh^2(z)$',
        'sech_identity.png',
        fontsize=20,
        figsize=(8, 1.2)
    )

    # 6. Lyapunov exponent definition
    print("6. Lyapunov exponent...")
    render_equation(
        r'$\lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{k=0}^{n-1} \ln \|J(\mathbf{x}_k)\|$',
        'lyapunov_exponent.png',
        fontsize=22,
        figsize=(10, 1.8),
        box=True,
        box_color='#e8e8f4'
    )

    # 7. Lyapunov spectrum
    print("7. Lyapunov spectrum...")
    render_equation(
        r'$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$',
        'lyapunov_spectrum.png',
        fontsize=22,
        figsize=(8, 1.2)
    )

    # 8. Fractal dimension
    print("8. Fractal dimension...")
    render_equation(
        r'$D_f = \lim_{\epsilon \to 0} \frac{\ln N(\epsilon)}{\ln(1/\epsilon)}$',
        'fractal_dimension.png',
        fontsize=22,
        figsize=(10, 1.8)
    )

    # 9. Hausdorff dimension condition
    print("9. Hausdorff dimension...")
    render_equation(
        r'$\dim_H(\partial\mathcal{B}) > d - 1$',
        'hausdorff_condition.png',
        fontsize=22,
        figsize=(8, 1.5)
    )

    # 10. Metastability scaling
    print("10. Metastability scaling...")
    render_equation(
        r'$\tau \propto \|\mathbf{x} - \partial\mathcal{B}\|^{-\alpha}$',
        'metastability.png',
        fontsize=22,
        figsize=(8, 1.5)
    )

    # 11. Stability condition
    print("11. Stability condition...")
    render_equation(
        r'$\rho(A) < 1$',
        'stability_condition.png',
        fontsize=22,
        figsize=(5, 1.2)
    )

    # 12. Attractor conditions - individual
    print("12. Attractor conditions...")

    # Fixed point
    render_equation(
        r'$\lambda_1 < 0 \quad \Rightarrow \quad \mathrm{Fixed\ Point}$',
        'attractor_fixed.png',
        fontsize=20,
        figsize=(10, 1.3)
    )

    # Limit cycle
    render_equation(
        r'$\lambda_1 = 0,\ \lambda_2 < 0 \quad \Rightarrow \quad \mathrm{Limit\ Cycle}$',
        'attractor_cycle.png',
        fontsize=20,
        figsize=(12, 1.3)
    )

    # Torus
    render_equation(
        r'$\lambda_1 = \lambda_2 = 0 \quad \Rightarrow \quad \mathrm{Torus}$',
        'attractor_torus.png',
        fontsize=20,
        figsize=(10, 1.3)
    )

    # Strange attractor
    render_equation(
        r'$\lambda_1 > 0 \quad \Rightarrow \quad \mathrm{Strange\ Attractor\ (Chaos)}$',
        'attractor_strange.png',
        fontsize=20,
        figsize=(12, 1.3)
    )

    print()
    print(f"All equation images saved to {OUTPUT_DIR}/")
    print()

    # List generated files
    files = sorted(os.listdir(OUTPUT_DIR))
    print("Generated files:")
    total_size = 0
    for f in files:
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
            total_size += size
            print(f"  {f}: {size/1024:.1f} KB")
    print(f"\nTotal: {total_size/1024:.1f} KB")


if __name__ == '__main__':
    main()

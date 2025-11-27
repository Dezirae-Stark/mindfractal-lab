/**
 * Quantum Gravity Viewer — Spacetime Weave Explorer
 * MindFractal Lab
 *
 * Visualizes emergent spacetime geometry from quantum foam fluctuations.
 */

import { MFViewerBase, COLORS } from './ui_common.js';

export class QuantumGravityViewer extends MFViewerBase {
    constructor(canvasId, controlsId) {
        super(canvasId, controlsId);

        this.params = {
            node_count: 100,
            noise_level: 0.3,
            coherence: 0.5,
            curvature_bias: 0.0,
            seed: 42
        };

        this.data = null;
        this.animating = false;
        this.animationFrame = 0;

        this.setupControls();
    }

    setupControls() {
        const controls = [
            { id: 'node-count', param: 'node_count', min: 25, max: 400, step: 25, label: 'Nodes' },
            { id: 'noise-level', param: 'noise_level', min: 0, max: 1, step: 0.05, label: 'Quantum Noise' },
            { id: 'coherence', param: 'coherence', min: 0, max: 1, step: 0.05, label: 'Coherence' },
            { id: 'curvature', param: 'curvature_bias', min: -1, max: 1, step: 0.1, label: 'Curvature' }
        ];

        controls.forEach(ctrl => {
            const slider = document.getElementById(ctrl.id);
            const display = document.getElementById(`${ctrl.id}-value`);

            if (slider) {
                slider.value = this.params[ctrl.param];
                if (display) display.textContent = this.params[ctrl.param];

                slider.addEventListener('input', () => {
                    this.params[ctrl.param] = parseFloat(slider.value);
                    if (display) display.textContent = slider.value;
                    this.compute();
                });
            }
        });

        // Animate button
        const animBtn = document.getElementById('animate-btn');
        if (animBtn) {
            animBtn.addEventListener('click', () => this.toggleAnimation());
        }

        // Randomize button
        const randBtn = document.getElementById('randomize-btn');
        if (randBtn) {
            randBtn.addEventListener('click', () => {
                this.params.seed = Math.floor(Math.random() * 10000);
                this.compute();
            });
        }
    }

    async compute() {
        if (!this.pyodide) {
            this.showStatus('Loading Python...', 'info');
            return;
        }

        this.showStatus('Computing spacetime weave...', 'info');

        try {
            const code = `
import json
from quantum_gravity_core import compute_spacetime_weave
result = compute_spacetime_weave(${JSON.stringify(this.params)})
json.dumps(result)
`;
            const resultJson = await this.pyodide.runPythonAsync(code);
            this.data = JSON.parse(resultJson);
            this.render();
            this.showStatus(`Weave: ${this.data.nodes.length} nodes, ${this.data.edges.length} edges`, 'success');
        } catch (err) {
            this.showStatus(`Error: ${err.message}`, 'error');
            console.error(err);
        }
    }

    render() {
        if (!this.data) return;

        const { ctx, width, height } = this.getContext();

        // Clear with dark background
        ctx.fillStyle = '#0a0a12';
        ctx.fillRect(0, 0, width, height);

        const { nodes, edges } = this.data;
        const scale = Math.min(width, height) * 0.95;
        const offsetX = (width - scale) / 2;
        const offsetY = (height - scale) / 2;

        // Draw edges first
        ctx.lineWidth = 1;
        edges.forEach(([i, j, weight]) => {
            const n1 = nodes[i];
            const n2 = nodes[j];
            if (!n1 || !n2) return;

            const x1 = offsetX + n1[0] * scale;
            const y1 = offsetY + n1[1] * scale;
            const x2 = offsetX + n2[0] * scale;
            const y2 = offsetY + n2[1] * scale;

            // Color based on weight
            const alpha = 0.2 + weight * 0.5;
            const hue = 200 + weight * 60; // Blue to cyan
            ctx.strokeStyle = `hsla(${hue}, 80%, 60%, ${alpha})`;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        });

        // Draw nodes
        nodes.forEach(([x, y, intensity]) => {
            const px = offsetX + x * scale;
            const py = offsetY + y * scale;
            const radius = 2 + intensity * 4;

            // Glow effect
            const gradient = ctx.createRadialGradient(px, py, 0, px, py, radius * 3);
            const hue = 180 + intensity * 40;
            gradient.addColorStop(0, `hsla(${hue}, 90%, 70%, ${intensity})`);
            gradient.addColorStop(0.5, `hsla(${hue}, 80%, 50%, ${intensity * 0.3})`);
            gradient.addColorStop(1, 'transparent');

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(px, py, radius * 3, 0, Math.PI * 2);
            ctx.fill();

            // Core
            ctx.fillStyle = `hsla(${hue}, 100%, 80%, ${intensity})`;
            ctx.beginPath();
            ctx.arc(px, py, radius, 0, Math.PI * 2);
            ctx.fill();
        });

        // Draw coherence indicator
        this.drawCoherenceIndicator(ctx, width, height);
    }

    drawCoherenceIndicator(ctx, width, height) {
        if (!this.data) return;

        const score = this.data.coherence_score;
        const barWidth = 100;
        const barHeight = 8;
        const x = width - barWidth - 15;
        const y = 15;

        // Background
        ctx.fillStyle = 'rgba(255,255,255,0.1)';
        ctx.fillRect(x, y, barWidth, barHeight);

        // Fill
        const hue = score * 120; // Red to green
        ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
        ctx.fillRect(x, y, barWidth * score, barHeight);

        // Label
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.font = '10px monospace';
        ctx.fillText(`Coherence: ${(score * 100).toFixed(0)}%`, x, y + barHeight + 12);
    }

    toggleAnimation() {
        this.animating = !this.animating;
        const btn = document.getElementById('animate-btn');
        if (btn) {
            btn.textContent = this.animating ? 'Stop' : 'Animate';
        }

        if (this.animating) {
            this.animate();
        }
    }

    async animate() {
        if (!this.animating) return;

        this.animationFrame++;

        try {
            const code = `
import json
from quantum_gravity_core import compute_foam_animation_frame
result = compute_foam_animation_frame(${JSON.stringify(this.params)}, ${this.animationFrame})
json.dumps(result)
`;
            const resultJson = await this.pyodide.runPythonAsync(code);
            this.data = JSON.parse(resultJson);
            this.render();
        } catch (err) {
            console.error(err);
        }

        if (this.animating) {
            setTimeout(() => this.animate(), 100);
        }
    }

    async initPyodide() {
        await super.initPyodide();

        // Load the quantum gravity core module
        try {
            const response = await fetch('../py/quantum_gravity_core.py');
            const code = await response.text();
            await this.pyodide.runPythonAsync(code);
            this.compute();
        } catch (err) {
            this.showStatus(`Failed to load module: ${err.message}`, 'error');
        }
    }
}

// Auto-initialize if canvas exists
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('quantum-gravity-canvas');
    if (canvas) {
        window.quantumGravityViewer = new QuantumGravityViewer('quantum-gravity-canvas', 'qg-controls');
        window.quantumGravityViewer.initPyodide();
    }
});

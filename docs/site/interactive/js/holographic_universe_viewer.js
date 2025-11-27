/**
 * Holographic Universe Viewer — Implicate/Explicate Projector
 * MindFractal Lab
 *
 * Visualizes holographic projection from boundary encodings to bulk patterns.
 */

import { MFViewerBase, COLORS } from './ui_common.js';

export class HolographicUniverseViewer extends MFViewerBase {
    constructor(canvasId, controlsId) {
        super(canvasId, controlsId);

        this.params = {
            resolution: 60,
            encoding_type: 'wave',
            projection_depth: 0.5,
            smoothness: 0.3,
            seed: 42
        };

        this.data = null;
        this.viewMode = 'split'; // 'split', 'boundary', 'explicate', 'overlay'
        this.brushActive = false;
        this.brushValue = 1.0;

        this.setupControls();
        this.setupInteraction();
    }

    setupControls() {
        const controls = [
            { id: 'projection-depth', param: 'projection_depth', min: 0.1, max: 1, step: 0.05, label: 'Depth' },
            { id: 'smoothness', param: 'smoothness', min: 0, max: 1, step: 0.05, label: 'Smoothness' },
            { id: 'resolution', param: 'resolution', min: 30, max: 80, step: 10, label: 'Resolution' }
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

        // Encoding type buttons
        const encodings = ['stripes', 'noise', 'wave', 'spiral', 'checkerboard'];
        encodings.forEach(enc => {
            const btn = document.getElementById(`enc-${enc}`);
            if (btn) {
                btn.addEventListener('click', () => {
                    this.params.encoding_type = enc;
                    document.querySelectorAll('.encoding-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    this.compute();
                });

                if (enc === this.params.encoding_type) {
                    btn.classList.add('active');
                }
            }
        });

        // View mode buttons
        const viewModes = ['split', 'boundary', 'explicate', 'overlay'];
        viewModes.forEach(mode => {
            const btn = document.getElementById(`view-${mode}`);
            if (btn) {
                btn.addEventListener('click', () => {
                    this.viewMode = mode;
                    document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    this.render();
                });

                if (mode === this.viewMode) {
                    btn.classList.add('active');
                }
            }
        });

        // Brush controls
        const brushBtn = document.getElementById('brush-toggle');
        if (brushBtn) {
            brushBtn.addEventListener('click', () => {
                this.brushActive = !this.brushActive;
                brushBtn.classList.toggle('active', this.brushActive);
                brushBtn.textContent = this.brushActive ? 'Brush: ON' : 'Brush: OFF';
            });
        }

        const brushSlider = document.getElementById('brush-value');
        if (brushSlider) {
            brushSlider.addEventListener('input', () => {
                this.brushValue = parseFloat(brushSlider.value);
            });
        }

        // Randomize
        const randBtn = document.getElementById('randomize-btn');
        if (randBtn) {
            randBtn.addEventListener('click', () => {
                this.params.seed = Math.floor(Math.random() * 10000);
                this.compute();
            });
        }
    }

    setupInteraction() {
        let painting = false;

        this.canvas.addEventListener('mousedown', (e) => {
            if (this.brushActive) {
                painting = true;
                this.paint(e);
            }
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (painting && this.brushActive) {
                this.paint(e);
            }
        });

        this.canvas.addEventListener('mouseup', () => {
            painting = false;
        });

        this.canvas.addEventListener('mouseleave', () => {
            painting = false;
        });

        // Touch support
        this.canvas.addEventListener('touchstart', (e) => {
            if (this.brushActive) {
                e.preventDefault();
                painting = true;
                this.paint(e.touches[0]);
            }
        });

        this.canvas.addEventListener('touchmove', (e) => {
            if (painting && this.brushActive) {
                e.preventDefault();
                this.paint(e.touches[0]);
            }
        });

        this.canvas.addEventListener('touchend', () => {
            painting = false;
        });
    }

    async paint(e) {
        if (!this.pyodide || !this.data) return;

        const rect = this.canvas.getBoundingClientRect();
        let x, y;

        if (this.viewMode === 'split') {
            // Only paint on left half (boundary)
            x = (e.clientX - rect.left) / (rect.width / 2);
            y = (e.clientY - rect.top) / rect.height;
            if (x > 1) return; // Clicked on right side
        } else {
            x = (e.clientX - rect.left) / rect.width;
            y = (e.clientY - rect.top) / rect.height;
        }

        x = Math.max(0, Math.min(1, x));
        y = Math.max(0, Math.min(1, y));

        try {
            const code = `
import json
from holographic_universe_core import compute_boundary_modification
result = compute_boundary_modification(${JSON.stringify(this.params)}, ${x}, ${y}, ${this.brushValue})
json.dumps(result)
`;
            const resultJson = await this.pyodide.runPythonAsync(code);
            this.data = JSON.parse(resultJson);
            this.render();
        } catch (err) {
            console.error(err);
        }
    }

    async compute() {
        if (!this.pyodide) {
            this.showStatus('Loading Python...', 'info');
            return;
        }

        this.showStatus('Computing holographic projection...', 'info');

        try {
            const code = `
import json
from holographic_universe_core import compute_implicate_explicate
result = compute_implicate_explicate(${JSON.stringify(this.params)})
json.dumps(result)
`;
            const resultJson = await this.pyodide.runPythonAsync(code);
            this.data = JSON.parse(resultJson);
            this.render();

            const entB = this.data.entropy_boundary.toFixed(2);
            const entE = this.data.entropy_explicate.toFixed(2);
            this.showStatus(`Entropy — Boundary: ${entB}, Bulk: ${entE}`, 'success');
        } catch (err) {
            this.showStatus(`Error: ${err.message}`, 'error');
            console.error(err);
        }
    }

    render() {
        if (!this.data) return;

        const { ctx, width, height } = this.getContext();

        // Clear
        ctx.fillStyle = '#0a0a10';
        ctx.fillRect(0, 0, width, height);

        switch (this.viewMode) {
            case 'split':
                this.renderSplit(ctx, width, height);
                break;
            case 'boundary':
                this.renderHeatmap(ctx, this.data.boundary, 0, 0, width, height, 'boundary');
                break;
            case 'explicate':
                this.renderHeatmap(ctx, this.data.explicate, 0, 0, width, height, 'explicate');
                break;
            case 'overlay':
                this.renderOverlay(ctx, width, height);
                break;
        }

        // Draw entropy comparison
        this.drawEntropyBars(ctx, width, height);
    }

    renderSplit(ctx, width, height) {
        const halfW = width / 2 - 2;

        // Boundary (left)
        this.renderHeatmap(ctx, this.data.boundary, 0, 0, halfW, height, 'boundary');

        // Divider
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.fillRect(halfW + 1, 0, 2, height);

        // Explicate (right)
        this.renderHeatmap(ctx, this.data.explicate, halfW + 4, 0, halfW, height, 'explicate');

        // Labels
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.font = '11px sans-serif';
        ctx.fillText('Boundary (Implicate)', 10, 20);
        ctx.fillText('Bulk (Explicate)', halfW + 14, 20);
    }

    renderHeatmap(ctx, grid, x0, y0, w, h, type) {
        const resolution = grid.length;
        const cellW = w / resolution;
        const cellH = h / resolution;

        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const value = grid[i][j];

                // Different color schemes for boundary vs explicate
                let hue, saturation, lightness;
                if (type === 'boundary') {
                    // Purple/magenta for boundary
                    hue = 280 + value * 40;
                    saturation = 60 + value * 20;
                    lightness = 15 + value * 45;
                } else {
                    // Cyan/blue for explicate
                    hue = 180 + value * 40;
                    saturation = 50 + value * 30;
                    lightness = 10 + value * 50;
                }

                ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
                ctx.fillRect(x0 + j * cellW, y0 + i * cellH, cellW + 1, cellH + 1);
            }
        }
    }

    renderOverlay(ctx, width, height) {
        const { boundary, explicate } = this.data;
        const resolution = boundary.length;
        const cellW = width / resolution;
        const cellH = height / resolution;

        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const bVal = boundary[i][j];
                const eVal = explicate[i][j];

                // Blend boundary (red) and explicate (cyan)
                const r = Math.floor(bVal * 200);
                const g = Math.floor(eVal * 180);
                const b = Math.floor(eVal * 220);

                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.fillRect(j * cellW, i * cellH, cellW + 1, cellH + 1);
            }
        }

        // Label
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.font = '11px sans-serif';
        ctx.fillText('Overlay: Boundary + Bulk', 10, 20);
    }

    drawEntropyBars(ctx, width, height) {
        const barWidth = 80;
        const barHeight = 6;
        const x = width - barWidth - 15;
        const y = height - 50;

        const maxEntropy = Math.log2(20); // Max possible with 20 bins

        // Boundary entropy
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.font = '9px monospace';
        ctx.fillText('Boundary S', x, y - 2);

        ctx.fillStyle = 'rgba(255,255,255,0.1)';
        ctx.fillRect(x, y, barWidth, barHeight);

        ctx.fillStyle = 'hsl(280, 70%, 50%)';
        ctx.fillRect(x, y, barWidth * (this.data.entropy_boundary / maxEntropy), barHeight);

        // Explicate entropy
        const y2 = y + 18;
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.fillText('Bulk S', x, y2 - 2);

        ctx.fillStyle = 'rgba(255,255,255,0.1)';
        ctx.fillRect(x, y2, barWidth, barHeight);

        ctx.fillStyle = 'hsl(200, 70%, 50%)';
        ctx.fillRect(x, y2, barWidth * (this.data.entropy_explicate / maxEntropy), barHeight);
    }

    async initPyodide() {
        await super.initPyodide();

        try {
            const response = await fetch('../py/holographic_universe_core.py');
            const code = await response.text();
            await this.pyodide.runPythonAsync(code);
            this.compute();
        } catch (err) {
            this.showStatus(`Failed to load module: ${err.message}`, 'error');
        }
    }
}

// Auto-initialize
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('holographic-universe-canvas');
    if (canvas) {
        window.holographicUniverseViewer = new HolographicUniverseViewer('holographic-universe-canvas', 'hu-controls');
        window.holographicUniverseViewer.initPyodide();
    }
});

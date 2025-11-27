/**
 * Gateway Coherence Viewer — Coherence & Resonance Map
 * MindFractal Lab
 *
 * Visualizes hemispheric synchronization patterns using coupled oscillators.
 */

import { MFViewerBase, COLORS } from './ui_common.js';

export class GatewayCoherenceViewer extends MFViewerBase {
    constructor(canvasId, controlsId) {
        super(canvasId, controlsId);

        this.params = {
            freq_left: 1.0,
            freq_right: 1.0,
            phase_offset: 0.0,
            coupling: 0.5,
            noise: 0.1,
            steps: 500
        };

        this.resonanceParams = {
            resolution: 50,
            freq_left: 1.0,
            freq_right: 1.0,
            coupling: 0.5
        };

        this.data = null;
        this.resonanceData = null;
        this.viewMode = 'pattern'; // 'pattern' or 'field'
        this.animating = false;
        this.trailPoints = [];

        this.setupControls();
    }

    setupControls() {
        const controls = [
            { id: 'freq-left', param: 'freq_left', min: 0.5, max: 5, step: 0.1, label: 'Left Freq' },
            { id: 'freq-right', param: 'freq_right', min: 0.5, max: 5, step: 0.1, label: 'Right Freq' },
            { id: 'coupling', param: 'coupling', min: 0, max: 1, step: 0.05, label: 'Coupling' },
            { id: 'phase-offset', param: 'phase_offset', min: 0, max: 6.28, step: 0.1, label: 'Phase' },
            { id: 'noise', param: 'noise', min: 0, max: 0.5, step: 0.05, label: 'Noise' }
        ];

        controls.forEach(ctrl => {
            const slider = document.getElementById(ctrl.id);
            const display = document.getElementById(`${ctrl.id}-value`);

            if (slider) {
                slider.value = this.params[ctrl.param];
                if (display) display.textContent = this.params[ctrl.param].toFixed(2);

                slider.addEventListener('input', () => {
                    this.params[ctrl.param] = parseFloat(slider.value);
                    this.resonanceParams[ctrl.param] = parseFloat(slider.value);
                    if (display) display.textContent = parseFloat(slider.value).toFixed(2);
                    this.compute();
                });
            }
        });

        // View mode toggle
        const patternBtn = document.getElementById('view-pattern');
        const fieldBtn = document.getElementById('view-field');

        if (patternBtn) {
            patternBtn.addEventListener('click', () => {
                this.viewMode = 'pattern';
                patternBtn.classList.add('active');
                fieldBtn?.classList.remove('active');
                this.compute();
            });
        }

        if (fieldBtn) {
            fieldBtn.addEventListener('click', () => {
                this.viewMode = 'field';
                fieldBtn.classList.add('active');
                patternBtn?.classList.remove('active');
                this.computeResonance();
            });
        }

        // Animation
        const animBtn = document.getElementById('animate-btn');
        if (animBtn) {
            animBtn.addEventListener('click', () => this.toggleAnimation());
        }

        // Binaural presets
        const presetBtns = document.querySelectorAll('.binaural-preset');
        presetBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const preset = btn.dataset.preset;
                this.applyPreset(preset);
            });
        });
    }

    applyPreset(preset) {
        const presets = {
            'delta': { freq_left: 1.0, freq_right: 1.0, coupling: 0.8 },
            'theta': { freq_left: 1.5, freq_right: 1.5, coupling: 0.6 },
            'alpha': { freq_left: 2.0, freq_right: 2.0, coupling: 0.5 },
            'beta': { freq_left: 3.0, freq_right: 3.0, coupling: 0.4 },
            'gamma': { freq_left: 4.0, freq_right: 4.0, coupling: 0.3 }
        };

        const p = presets[preset];
        if (p) {
            Object.assign(this.params, p);
            Object.assign(this.resonanceParams, p);

            // Update UI
            ['freq-left', 'freq-right', 'coupling'].forEach(id => {
                const slider = document.getElementById(id);
                const param = id.replace('-', '_');
                if (slider && this.params[param] !== undefined) {
                    slider.value = this.params[param];
                    const display = document.getElementById(`${id}-value`);
                    if (display) display.textContent = this.params[param].toFixed(2);
                }
            });

            this.compute();
        }
    }

    async compute() {
        if (!this.pyodide) {
            this.showStatus('Loading Python...', 'info');
            return;
        }

        this.showStatus('Computing coherence pattern...', 'info');

        try {
            const code = `
import json
from gateway_coherence_core import compute_coherence_pattern
result = compute_coherence_pattern(${JSON.stringify(this.params)})
json.dumps(result)
`;
            const resultJson = await this.pyodide.runPythonAsync(code);
            this.data = JSON.parse(resultJson);
            this.render();

            const lockStatus = this.data.phase_lock ? 'LOCKED' : 'unlocked';
            this.showStatus(`Coherence: ${(this.data.coherence_metric * 100).toFixed(0)}% — Phase ${lockStatus}`, 'success');
        } catch (err) {
            this.showStatus(`Error: ${err.message}`, 'error');
            console.error(err);
        }
    }

    async computeResonance() {
        if (!this.pyodide) return;

        this.showStatus('Computing resonance field...', 'info');

        try {
            const code = `
import json
from gateway_coherence_core import compute_resonance_field
result = compute_resonance_field(${JSON.stringify(this.resonanceParams)})
json.dumps(result)
`;
            const resultJson = await this.pyodide.runPythonAsync(code);
            this.resonanceData = JSON.parse(resultJson);
            this.renderResonanceField();
            this.showStatus(`Resonance field: ${this.resonanceData.peak_locations.length} peaks`, 'success');
        } catch (err) {
            this.showStatus(`Error: ${err.message}`, 'error');
        }
    }

    render() {
        if (this.viewMode === 'field') {
            this.renderResonanceField();
            return;
        }

        if (!this.data) return;

        const { ctx, width, height } = this.getContext();

        // Dark background
        ctx.fillStyle = '#0a0a18';
        ctx.fillRect(0, 0, width, height);

        const { points, coherence_metric, phase_lock } = this.data;

        // Draw trail
        if (points.length > 0) {
            ctx.beginPath();
            ctx.moveTo(points[0][0] * width, points[0][1] * height);

            for (let i = 1; i < points.length; i++) {
                const [x, y, intensity] = points[i];
                const px = x * width;
                const py = y * height;

                // Color based on intensity and position in trail
                const progress = i / points.length;
                const alpha = 0.1 + intensity * 0.6 * progress;
                const hue = phase_lock ? 160 : 280 - progress * 60;

                ctx.strokeStyle = `hsla(${hue}, 70%, 60%, ${alpha})`;
                ctx.lineWidth = 1 + intensity * 2;
                ctx.lineTo(px, py);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(px, py);
            }
        }

        // Draw current position (last few points)
        const recentPoints = points.slice(-20);
        recentPoints.forEach((point, i) => {
            const [x, y, intensity] = point;
            const px = x * width;
            const py = y * height;
            const progress = i / recentPoints.length;

            const radius = 2 + intensity * 4 * progress;
            const hue = phase_lock ? 160 : 280;

            ctx.fillStyle = `hsla(${hue}, 80%, ${50 + intensity * 30}%, ${progress})`;
            ctx.beginPath();
            ctx.arc(px, py, radius, 0, Math.PI * 2);
            ctx.fill();
        });

        // Draw coherence indicator
        this.drawCoherenceIndicator(ctx, width, height, coherence_metric, phase_lock);
    }

    renderResonanceField() {
        if (!this.resonanceData) return;

        const { ctx, width, height } = this.getContext();
        const { field, peak_locations } = this.resonanceData;
        const resolution = field.length;

        // Draw heatmap
        const cellW = width / resolution;
        const cellH = height / resolution;

        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const value = field[i][j];
                const hue = 200 + value * 60;
                const lightness = 10 + value * 50;

                ctx.fillStyle = `hsl(${hue}, 70%, ${lightness}%)`;
                ctx.fillRect(j * cellW, i * cellH, cellW + 1, cellH + 1);
            }
        }

        // Mark peaks
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        peak_locations.forEach(([x, y]) => {
            const px = x * width;
            const py = y * height;

            ctx.beginPath();
            ctx.arc(px, py, 3, 0, Math.PI * 2);
            ctx.fill();
        });

        // Label
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.font = '11px monospace';
        ctx.fillText('Resonance Field', 10, 20);
    }

    drawCoherenceIndicator(ctx, width, height, coherence, phaseLock) {
        const x = 10;
        const y = 10;
        const barWidth = 100;
        const barHeight = 8;

        // Background
        ctx.fillStyle = 'rgba(255,255,255,0.1)';
        ctx.fillRect(x, y, barWidth, barHeight);

        // Fill
        const hue = phaseLock ? 140 : 280;
        ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
        ctx.fillRect(x, y, barWidth * coherence, barHeight);

        // Labels
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.font = '10px monospace';
        ctx.fillText(`Coherence: ${(coherence * 100).toFixed(0)}%`, x, y + barHeight + 12);

        if (phaseLock) {
            ctx.fillStyle = 'rgba(100, 255, 150, 0.8)';
            ctx.fillText('PHASE LOCKED', x, y + barHeight + 24);
        }
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

    animate() {
        if (!this.animating) return;

        // Slowly shift phase
        this.params.phase_offset += 0.05;
        if (this.params.phase_offset > Math.PI * 2) {
            this.params.phase_offset = 0;
        }

        const slider = document.getElementById('phase-offset');
        if (slider) slider.value = this.params.phase_offset;

        this.compute().then(() => {
            if (this.animating) {
                requestAnimationFrame(() => this.animate());
            }
        });
    }

    async initPyodide() {
        await super.initPyodide();

        try {
            const response = await fetch('../py/gateway_coherence_core.py');
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
    const canvas = document.getElementById('gateway-coherence-canvas');
    if (canvas) {
        window.gatewayCoherenceViewer = new GatewayCoherenceViewer('gateway-coherence-canvas', 'gc-controls');
        window.gatewayCoherenceViewer.initPyodide();
    }
});

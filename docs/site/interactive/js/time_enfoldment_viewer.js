/**
 * Time Enfoldment Viewer — Bidirectional Timeline Navigator
 * MindFractal Lab
 *
 * Visualizes past-future bidirectional time structures with retrocausal influences.
 */

import { MFViewerBase, COLORS } from './ui_common.js';

export class TimeEnfoldmentViewer extends MFViewerBase {
    constructor(canvasId, controlsId) {
        super(canvasId, controlsId);

        this.params = {
            depth: 4,
            branching_factor: 2.0,
            retro_weight: 0.3,
            decoherence: 0.2,
            seed: 42
        };

        this.data = null;
        this.decisionStrength = 0;
        this.hoveredNode = null;

        this.setupControls();
        this.setupInteraction();
    }

    setupControls() {
        const controls = [
            { id: 'depth', param: 'depth', min: 1, max: 8, step: 1, label: 'Depth' },
            { id: 'branching', param: 'branching_factor', min: 1, max: 4, step: 0.5, label: 'Branching' },
            { id: 'retro-weight', param: 'retro_weight', min: 0, max: 1, step: 0.1, label: 'Retrocausal' },
            { id: 'decoherence', param: 'decoherence', min: 0, max: 1, step: 0.1, label: 'Decoherence' }
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

        // Decision slider (affects timeline reshaping)
        const decisionSlider = document.getElementById('decision-strength');
        if (decisionSlider) {
            decisionSlider.addEventListener('input', () => {
                this.decisionStrength = parseFloat(decisionSlider.value);
                const display = document.getElementById('decision-strength-value');
                if (display) display.textContent = this.decisionStrength.toFixed(1);
                this.computeDecision();
            });
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

    setupInteraction() {
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;
            this.checkHover(x, y);
        });

        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;
            this.handleClick(x, y);
        });
    }

    checkHover(x, y) {
        if (!this.data) return;

        const threshold = 0.03;
        this.hoveredNode = null;

        // Check past nodes
        for (let i = 0; i < this.data.past_nodes.length; i++) {
            const [nx, ny] = this.data.past_nodes[i];
            if (Math.abs(x - nx) < threshold && Math.abs(y - ny) < threshold) {
                this.hoveredNode = { type: 'past', index: i, node: this.data.past_nodes[i] };
                break;
            }
        }

        // Check future nodes
        if (!this.hoveredNode) {
            for (let i = 0; i < this.data.future_nodes.length; i++) {
                const [nx, ny] = this.data.future_nodes[i];
                if (Math.abs(x - nx) < threshold && Math.abs(y - ny) < threshold) {
                    this.hoveredNode = { type: 'future', index: i, node: this.data.future_nodes[i] };
                    break;
                }
            }
        }

        this.render();
    }

    handleClick(x, y) {
        // Click near center to reset decision
        const [px, py] = this.data?.present || [0.5, 0.5];
        if (Math.abs(x - px) < 0.05 && Math.abs(y - py) < 0.05) {
            const decisionSlider = document.getElementById('decision-strength');
            if (decisionSlider) {
                decisionSlider.value = 0;
                this.decisionStrength = 0;
                this.compute();
            }
        }
    }

    async compute() {
        if (!this.pyodide) {
            this.showStatus('Loading Python...', 'info');
            return;
        }

        this.showStatus('Computing timeline...', 'info');

        try {
            const code = `
import json
from time_enfoldment_core import compute_time_enfoldment
result = compute_time_enfoldment(${JSON.stringify(this.params)})
json.dumps(result)
`;
            const resultJson = await this.pyodide.runPythonAsync(code);
            this.data = JSON.parse(resultJson);
            this.render();
            this.showStatus(`Timeline: ${this.data.past_count} past, ${this.data.future_count} future branches`, 'success');
        } catch (err) {
            this.showStatus(`Error: ${err.message}`, 'error');
            console.error(err);
        }
    }

    async computeDecision() {
        if (!this.pyodide || this.decisionStrength === 0) {
            this.compute();
            return;
        }

        try {
            const code = `
import json
from time_enfoldment_core import compute_decision_impact
result = compute_decision_impact(${JSON.stringify(this.params)}, ${this.decisionStrength})
json.dumps(result)
`;
            const resultJson = await this.pyodide.runPythonAsync(code);
            this.data = JSON.parse(resultJson);
            this.render();
        } catch (err) {
            console.error(err);
        }
    }

    render() {
        if (!this.data) return;

        const { ctx, width, height } = this.getContext();

        // Clear with dark gradient
        const gradient = ctx.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, '#1a0a20');    // Past: purple tint
        gradient.addColorStop(0.5, '#0a0a15');  // Present: dark
        gradient.addColorStop(1, '#0a1520');    // Future: blue tint
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);

        const { present, past_nodes, future_nodes, edges } = this.data;

        // Draw edges
        this.drawEdges(ctx, width, height, edges, past_nodes, future_nodes);

        // Draw past nodes
        past_nodes.forEach((node, i) => {
            const isHovered = this.hoveredNode?.type === 'past' && this.hoveredNode?.index === i;
            this.drawTimeNode(ctx, node, width, height, 'past', isHovered);
        });

        // Draw future nodes
        future_nodes.forEach((node, i) => {
            const isHovered = this.hoveredNode?.type === 'future' && this.hoveredNode?.index === i;
            this.drawTimeNode(ctx, node, width, height, 'future', isHovered);
        });

        // Draw present node (center)
        this.drawPresentNode(ctx, present, width, height);

        // Draw retrocausal indicator
        this.drawRetroIndicator(ctx, width, height);
    }

    drawEdges(ctx, width, height, edges, pastNodes, futureNodes) {
        edges.forEach(([fromIdx, toIdx, direction]) => {
            let x1, y1, x2, y2;

            if (direction === 'past') {
                if (fromIdx === -1) {
                    // From present
                    x1 = 0.5 * width;
                    y1 = 0.5 * height;
                } else {
                    const node = pastNodes[fromIdx];
                    x1 = node[0] * width;
                    y1 = node[1] * height;
                }
                const toNode = pastNodes[toIdx];
                if (!toNode) return;
                x2 = toNode[0] * width;
                y2 = toNode[1] * height;

                // Purple gradient for past
                const grad = ctx.createLinearGradient(x1, y1, x2, y2);
                grad.addColorStop(0, 'rgba(180, 100, 255, 0.6)');
                grad.addColorStop(1, 'rgba(120, 60, 200, 0.3)');
                ctx.strokeStyle = grad;
            } else {
                if (fromIdx === -1) {
                    x1 = 0.5 * width;
                    y1 = 0.5 * height;
                } else {
                    const nodeIdx = fromIdx - pastNodes.length;
                    const node = futureNodes[nodeIdx];
                    if (!node) return;
                    x1 = node[0] * width;
                    y1 = node[1] * height;
                }
                const toNode = futureNodes[toIdx - pastNodes.length];
                if (!toNode) return;
                x2 = toNode[0] * width;
                y2 = toNode[1] * height;

                // Cyan gradient for future
                const grad = ctx.createLinearGradient(x1, y1, x2, y2);
                grad.addColorStop(0, 'rgba(100, 200, 255, 0.6)');
                grad.addColorStop(1, 'rgba(60, 150, 220, 0.3)');
                ctx.strokeStyle = grad;
            }

            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        });
    }

    drawTimeNode(ctx, node, width, height, type, isHovered) {
        const [x, y, level, weight] = node;
        const px = x * width;
        const py = y * height;
        const radius = 4 + weight * 8;

        // Color based on type
        const hue = type === 'past' ? 280 : 200; // Purple for past, cyan for future
        const lightness = 50 + weight * 20;

        // Glow
        const glow = ctx.createRadialGradient(px, py, 0, px, py, radius * 3);
        glow.addColorStop(0, `hsla(${hue}, 70%, ${lightness}%, ${weight})`);
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(px, py, radius * 3, 0, Math.PI * 2);
        ctx.fill();

        // Core
        ctx.fillStyle = isHovered
            ? `hsl(${hue}, 100%, 80%)`
            : `hsla(${hue}, 80%, ${lightness}%, ${0.5 + weight * 0.5})`;
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();

        // Hover info
        if (isHovered) {
            ctx.fillStyle = 'rgba(255,255,255,0.9)';
            ctx.font = '11px monospace';
            const label = `${type === 'past' ? 'Past' : 'Future'} L${level} (${(weight * 100).toFixed(0)}%)`;
            ctx.fillText(label, px + radius + 5, py + 4);
        }
    }

    drawPresentNode(ctx, present, width, height) {
        const px = present[0] * width;
        const py = present[1] * height;
        const radius = 12;

        // Bright glow
        const glow = ctx.createRadialGradient(px, py, 0, px, py, radius * 4);
        glow.addColorStop(0, 'rgba(255, 255, 255, 1)');
        glow.addColorStop(0.3, 'rgba(255, 220, 100, 0.6)');
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(px, py, radius * 4, 0, Math.PI * 2);
        ctx.fill();

        // Core
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();

        // Label
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        ctx.font = 'bold 12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('NOW', px, py + radius + 18);
        ctx.textAlign = 'left';
    }

    drawRetroIndicator(ctx, width, height) {
        const retroInfluence = this.data.retro_influence;
        const barWidth = 80;
        const barHeight = 6;
        const x = 15;
        const y = height - 25;

        // Label
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.font = '10px monospace';
        ctx.fillText('Retrocausal Influence', x, y - 5);

        // Background
        ctx.fillStyle = 'rgba(255,255,255,0.1)';
        ctx.fillRect(x, y, barWidth, barHeight);

        // Fill
        ctx.fillStyle = `hsl(${280 - retroInfluence * 80}, 70%, 50%)`;
        ctx.fillRect(x, y, barWidth * retroInfluence, barHeight);
    }

    async initPyodide() {
        await super.initPyodide();

        try {
            const response = await fetch('../py/time_enfoldment_core.py');
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
    const canvas = document.getElementById('time-enfoldment-canvas');
    if (canvas) {
        window.timeEnfoldmentViewer = new TimeEnfoldmentViewer('time-enfoldment-canvas', 'te-controls');
        window.timeEnfoldmentViewer.initPyodide();
    }
});

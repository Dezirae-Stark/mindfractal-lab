/**
 * Many Worlds Viewer — Branching Universe Explorer
 * MindFractal Lab
 *
 * Visualizes quantum branching and decoherence in a many-worlds interpretation.
 */

import { MFViewerBase, COLORS } from './ui_common.js';

export class ManyWorldsViewer extends MFViewerBase {
    constructor(canvasId, controlsId) {
        super(canvasId, controlsId);

        this.params = {
            depth: 5,
            branching_factor: 2.0,
            decoherence: 0.1,
            prob_compression: 0.3,
            seed: 42
        };

        this.interferenceParams = {
            n_branches: 4,
            coherence: 0.5,
            resolution: 60,
            seed: 42
        };

        this.data = null;
        this.interferenceData = null;
        this.viewMode = 'tree'; // 'tree' or 'interference'
        this.selectedPath = [];
        this.hoveredNode = null;

        this.setupControls();
        this.setupInteraction();
    }

    setupControls() {
        const controls = [
            { id: 'depth', param: 'depth', min: 2, max: 8, step: 1, label: 'Depth' },
            { id: 'branching', param: 'branching_factor', min: 1.5, max: 4, step: 0.25, label: 'Branching' },
            { id: 'decoherence', param: 'decoherence', min: 0, max: 0.5, step: 0.05, label: 'Decoherence' },
            { id: 'prob-compression', param: 'prob_compression', min: 0, max: 0.8, step: 0.1, label: 'Compression' }
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

        // Interference controls
        const intControls = [
            { id: 'n-branches', param: 'n_branches', min: 2, max: 8, step: 1 },
            { id: 'int-coherence', param: 'coherence', min: 0, max: 1, step: 0.1 }
        ];

        intControls.forEach(ctrl => {
            const slider = document.getElementById(ctrl.id);
            const display = document.getElementById(`${ctrl.id}-value`);

            if (slider) {
                slider.value = this.interferenceParams[ctrl.param];
                if (display) display.textContent = this.interferenceParams[ctrl.param];

                slider.addEventListener('input', () => {
                    this.interferenceParams[ctrl.param] = parseFloat(slider.value);
                    if (display) display.textContent = slider.value;
                    if (this.viewMode === 'interference') {
                        this.computeInterference();
                    }
                });
            }
        });

        // View mode toggle
        const treeBtn = document.getElementById('view-tree');
        const intBtn = document.getElementById('view-interference');

        if (treeBtn) {
            treeBtn.addEventListener('click', () => {
                this.viewMode = 'tree';
                treeBtn.classList.add('active');
                intBtn?.classList.remove('active');
                this.render();
            });
        }

        if (intBtn) {
            intBtn.addEventListener('click', () => {
                this.viewMode = 'interference';
                intBtn.classList.add('active');
                treeBtn?.classList.remove('active');
                this.computeInterference();
            });
        }

        // Randomize
        const randBtn = document.getElementById('randomize-btn');
        if (randBtn) {
            randBtn.addEventListener('click', () => {
                this.params.seed = Math.floor(Math.random() * 10000);
                this.interferenceParams.seed = this.params.seed;
                this.selectedPath = [];
                this.compute();
            });
        }

        // Clear selection
        const clearBtn = document.getElementById('clear-selection');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.selectedPath = [];
                this.compute();
            });
        }
    }

    setupInteraction() {
        this.canvas.addEventListener('mousemove', (e) => {
            if (this.viewMode !== 'tree' || !this.data) return;

            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;

            this.checkNodeHover(x, y);
        });

        this.canvas.addEventListener('click', (e) => {
            if (this.viewMode !== 'tree' || !this.data) return;

            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;

            this.selectNode(x, y);
        });
    }

    checkNodeHover(mx, my) {
        const threshold = 0.03;
        this.hoveredNode = null;

        for (const node of this.data.nodes) {
            const [id, level, weight, x, y] = node;
            if (Math.abs(mx - x) < threshold && Math.abs(my - y) < threshold) {
                this.hoveredNode = { id, level, weight, x, y };
                break;
            }
        }

        this.render();
    }

    selectNode(mx, my) {
        const threshold = 0.03;

        for (const node of this.data.nodes) {
            const [id, level, weight, x, y] = node;
            if (Math.abs(mx - x) < threshold && Math.abs(my - y) < threshold) {
                // Build path from root to this node
                this.buildPathTo(id);
                this.render();
                break;
            }
        }
    }

    buildPathTo(targetId) {
        // Find path from root (id 0) to target
        const nodeMap = new Map();
        this.data.nodes.forEach(n => nodeMap.set(n[0], n));

        const parentMap = new Map();
        this.data.edges.forEach(([from, to]) => {
            parentMap.set(to, from);
        });

        const path = [targetId];
        let current = targetId;
        while (parentMap.has(current)) {
            current = parentMap.get(current);
            path.unshift(current);
        }

        this.selectedPath = path;
    }

    async compute() {
        if (!this.pyodide) {
            this.showStatus('Loading Python...', 'info');
            return;
        }

        this.showStatus('Computing branching universe...', 'info');

        try {
            let code;
            if (this.selectedPath.length > 0) {
                code = `
import json
from many_worlds_core import compute_branch_selection
result = compute_branch_selection(${JSON.stringify(this.params)}, ${JSON.stringify(this.selectedPath)})
json.dumps(result)
`;
            } else {
                code = `
import json
from many_worlds_core import compute_branching_universe
result = compute_branching_universe(${JSON.stringify(this.params)})
json.dumps(result)
`;
            }

            const resultJson = await this.pyodide.runPythonAsync(code);
            this.data = JSON.parse(resultJson);
            this.render();

            const probSum = (this.data.probability_sum * 100).toFixed(1);
            this.showStatus(`${this.data.total_branches} branches, ${probSum}% total probability`, 'success');
        } catch (err) {
            this.showStatus(`Error: ${err.message}`, 'error');
            console.error(err);
        }
    }

    async computeInterference() {
        if (!this.pyodide) return;

        this.showStatus('Computing interference pattern...', 'info');

        try {
            const code = `
import json
from many_worlds_core import compute_interference_pattern
result = compute_interference_pattern(${JSON.stringify(this.interferenceParams)})
json.dumps(result)
`;
            const resultJson = await this.pyodide.runPythonAsync(code);
            this.interferenceData = JSON.parse(resultJson);
            this.renderInterference();

            const vis = (this.interferenceData.visibility * 100).toFixed(0);
            this.showStatus(`Interference visibility: ${vis}%`, 'success');
        } catch (err) {
            this.showStatus(`Error: ${err.message}`, 'error');
        }
    }

    render() {
        if (this.viewMode === 'interference') {
            this.renderInterference();
            return;
        }

        if (!this.data) return;

        const { ctx, width, height } = this.getContext();

        // Dark background
        ctx.fillStyle = '#08080f';
        ctx.fillRect(0, 0, width, height);

        const { nodes, edges, highlighted_edges } = this.data;
        const highlightedEdgeSet = new Set(
            (highlighted_edges || []).map(e => `${e[0]}-${e[1]}`)
        );
        const selectedSet = new Set(this.selectedPath);

        // Draw edges
        ctx.lineWidth = 1;
        edges.forEach(([from, to]) => {
            const fromNode = nodes.find(n => n[0] === from);
            const toNode = nodes.find(n => n[0] === to);
            if (!fromNode || !toNode) return;

            const x1 = fromNode[3] * width;
            const y1 = fromNode[4] * height;
            const x2 = toNode[3] * width;
            const y2 = toNode[4] * height;

            const isHighlighted = highlightedEdgeSet.has(`${from}-${to}`);

            if (isHighlighted) {
                ctx.strokeStyle = 'rgba(255, 220, 100, 0.9)';
                ctx.lineWidth = 3;
            } else {
                const weight = toNode[2];
                const alpha = 0.15 + weight * 0.4;
                ctx.strokeStyle = `rgba(100, 180, 255, ${alpha})`;
                ctx.lineWidth = 1;
            }

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        });

        // Draw nodes
        nodes.forEach(node => {
            const [id, level, weight, x, y] = node;
            const px = x * width;
            const py = y * height;

            const isSelected = selectedSet.has(id);
            const isHovered = this.hoveredNode?.id === id;
            const isRoot = id === 0;

            let radius = 3 + weight * 10;
            let hue = 200 + level * 15;
            let lightness = 40 + weight * 30;

            if (isRoot) {
                radius = 10;
                hue = 60;
                lightness = 70;
            }

            if (isSelected) {
                hue = 45;
                lightness = 65;
            }

            // Glow
            const glow = ctx.createRadialGradient(px, py, 0, px, py, radius * 2.5);
            glow.addColorStop(0, `hsla(${hue}, 70%, ${lightness}%, ${weight})`);
            glow.addColorStop(1, 'transparent');
            ctx.fillStyle = glow;
            ctx.beginPath();
            ctx.arc(px, py, radius * 2.5, 0, Math.PI * 2);
            ctx.fill();

            // Core
            ctx.fillStyle = isHovered
                ? `hsl(${hue}, 100%, 80%)`
                : `hsla(${hue}, 80%, ${lightness}%, ${0.6 + weight * 0.4})`;
            ctx.beginPath();
            ctx.arc(px, py, radius, 0, Math.PI * 2);
            ctx.fill();

            // Probability label for selected nodes
            if (isSelected && !isRoot) {
                ctx.fillStyle = 'rgba(255, 220, 100, 0.9)';
                ctx.font = '10px monospace';
                ctx.fillText(`${(weight * 100).toFixed(1)}%`, px + radius + 3, py + 3);
            }
        });

        // Hover tooltip
        if (this.hoveredNode) {
            const { id, level, weight, x, y } = this.hoveredNode;
            const px = x * width;
            const py = y * height;

            ctx.fillStyle = 'rgba(0,0,0,0.8)';
            ctx.fillRect(px + 15, py - 25, 100, 40);

            ctx.fillStyle = 'rgba(255,255,255,0.9)';
            ctx.font = '10px monospace';
            ctx.fillText(`Branch ${id}`, px + 20, py - 10);
            ctx.fillText(`Level: ${level}`, px + 20, py + 2);
            ctx.fillText(`Prob: ${(weight * 100).toFixed(2)}%`, px + 20, py + 14);
        }

        // Draw legend
        this.drawLegend(ctx, width, height);
    }

    renderInterference() {
        if (!this.interferenceData) return;

        const { ctx, width, height } = this.getContext();
        const { pattern, visibility } = this.interferenceData;
        const resolution = pattern.length;
        const cellW = width / resolution;
        const cellH = height / resolution;

        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const value = pattern[i][j];
                const hue = 220 + value * 40;
                const lightness = value * 60;

                ctx.fillStyle = `hsl(${hue}, 70%, ${lightness}%)`;
                ctx.fillRect(j * cellW, i * cellH, cellW + 1, cellH + 1);
            }
        }

        // Visibility indicator
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.font = '11px monospace';
        ctx.fillText(`Fringe Visibility: ${(visibility * 100).toFixed(0)}%`, 10, 20);
        ctx.fillText(`${this.interferenceData.n_branches} branches`, 10, 35);
        ctx.fillText(`Coherence: ${(this.interferenceData.coherence * 100).toFixed(0)}%`, 10, 50);
    }

    drawLegend(ctx, width, height) {
        const x = 10;
        const y = height - 60;

        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(x - 5, y - 15, 140, 55);

        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.font = '10px monospace';
        ctx.fillText(`Branches: ${this.data.total_branches}`, x, y);
        ctx.fillText(`Max Depth: ${this.data.max_depth}`, x, y + 13);
        ctx.fillText(`Prob Sum: ${(this.data.probability_sum * 100).toFixed(1)}%`, x, y + 26);

        if (this.selectedPath.length > 0) {
            const pathProb = this.data.path_probability || 0;
            ctx.fillStyle = 'rgba(255, 220, 100, 0.9)';
            ctx.fillText(`Selected: ${(pathProb * 100).toFixed(2)}%`, x, y + 39);
        }
    }

    async initPyodide() {
        await super.initPyodide();

        try {
            const response = await fetch('../py/many_worlds_core.py');
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
    const canvas = document.getElementById('many-worlds-canvas');
    if (canvas) {
        window.manyWorldsViewer = new ManyWorldsViewer('many-worlds-canvas', 'mw-controls');
        window.manyWorldsViewer.initPyodide();
    }
});

/**
 * Child Mind Lab Viewer
 * MindFractal Lab
 *
 * Interactive visualization of Child Mind v1 synthetic agent.
 * Displays manifold state z_t projected to 2D, with coherence as color/radius.
 * Provides controls for curiosity, coherence preference, and stepping.
 */

(function() {
    'use strict';

    // Module state
    let pyodide = null;
    let isInitialized = false;
    let isRunning = false;
    let animationId = null;

    // Current state data
    let currentState = null;
    let trajectoryHistory = [];
    const MAX_HISTORY = 200;

    // Canvas and context
    let canvas = null;
    let ctx = null;

    // UI elements
    let curiositySlider = null;
    let coherenceSlider = null;
    let stepButton = null;
    let autoRunButton = null;
    let resetButton = null;
    let coherenceDisplay = null;
    let rewardDisplay = null;
    let stabilityDisplay = null;
    let noveltyDisplay = null;
    let stepCountDisplay = null;
    let statusLabel = null;

    // Color palette for coherence visualization
    const COLORS = {
        background: '#0a0a1a',
        gridLines: 'rgba(107, 107, 255, 0.1)',
        gridCenter: 'rgba(107, 107, 255, 0.3)',
        trajectoryLow: '#ff4fa3',      // Low coherence - pink
        trajectoryMid: '#c45aff',       // Mid coherence - purple
        trajectoryHigh: '#39d8ff',      // High coherence - cyan
        currentPoint: '#ffffff',
        textPrimary: '#e8e8ff',
        glowColor: 'rgba(107, 107, 255, 0.5)'
    };

    /**
     * Initialize the Child Mind viewer
     */
    async function initChildMind() {
        console.log('[ChildMind] Initializing...');

        // Get canvas
        canvas = document.getElementById('child-mind-canvas');
        if (!canvas) {
            console.error('[ChildMind] Canvas not found');
            return;
        }
        ctx = canvas.getContext('2d');

        // Get UI elements
        curiositySlider = document.getElementById('curiosity-slider');
        coherenceSlider = document.getElementById('coherence-pref-slider');
        stepButton = document.getElementById('step-btn');
        autoRunButton = document.getElementById('auto-run-btn');
        resetButton = document.getElementById('reset-btn');
        coherenceDisplay = document.getElementById('coherence-value');
        rewardDisplay = document.getElementById('reward-value');
        stabilityDisplay = document.getElementById('stability-value');
        noveltyDisplay = document.getElementById('novelty-value');
        stepCountDisplay = document.getElementById('step-count');
        statusLabel = document.getElementById('child-mind-status');

        // Setup canvas size
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        // Setup event listeners
        if (stepButton) {
            stepButton.addEventListener('click', stepOnce);
        }
        if (autoRunButton) {
            autoRunButton.addEventListener('click', toggleAutoRun);
        }
        if (resetButton) {
            resetButton.addEventListener('click', resetSimulation);
        }

        // Slider value displays
        if (curiositySlider) {
            curiositySlider.addEventListener('input', updateSliderDisplays);
        }
        if (coherenceSlider) {
            coherenceSlider.addEventListener('input', updateSliderDisplays);
        }

        // Initial render
        renderBackground();
        updateStatus('Loading Pyodide...');

        // Load Pyodide
        try {
            await loadPyodideAndInit();
            updateStatus('Ready');
            isInitialized = true;
        } catch (err) {
            console.error('[ChildMind] Failed to initialize:', err);
            updateStatus('Failed to load');
        }
    }

    /**
     * Load Pyodide and initialize the Child Mind module
     */
    async function loadPyodideAndInit() {
        // Check if Pyodide is already loaded globally
        if (typeof loadPyodide === 'undefined') {
            // Load Pyodide script
            await new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js';
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }

        updateStatus('Initializing Python...');
        pyodide = await loadPyodide();

        // Load numpy
        updateStatus('Loading NumPy...');
        await pyodide.loadPackage('numpy');

        // Fetch and run the Child Mind Pyodide module
        updateStatus('Loading Child Mind...');
        const response = await fetch('../py/child_mind_pyodide.py');
        const pyCode = await response.text();
        await pyodide.runPythonAsync(pyCode);

        // Initialize the simulation
        updateStatus('Resetting simulation...');
        const seed = Math.floor(Math.random() * 10000);
        const resultJson = pyodide.runPython(`reset_child_mind(${seed})`);
        const result = JSON.parse(resultJson);

        if (result.status === 'ok') {
            currentState = result.state;
            trajectoryHistory = [extractPoint(currentState)];
            updateDisplays();
            render();
        }
    }

    /**
     * Resize canvas to fit container
     */
    function resizeCanvas() {
        if (!canvas) return;
        const container = canvas.parentElement;
        const rect = container.getBoundingClientRect();
        const size = Math.min(rect.width - 20, 500);
        canvas.width = size;
        canvas.height = size;
        if (isInitialized) {
            render();
        } else {
            renderBackground();
        }
    }

    /**
     * Extract 2D point from state for visualization
     */
    function extractPoint(state) {
        // Use first two z components, map through tanh for bounded display
        const z = state.z || [0, 0];
        const x = Math.tanh(z[0] || 0);
        const y = Math.tanh(z[1] || 0);
        const coherence = state.summary?.coherence || 0.5;
        const stability = state.summary?.stability || 0.5;
        return { x, y, coherence, stability };
    }

    /**
     * Get color based on coherence value
     */
    function getCoherenceColor(coherence) {
        if (coherence < 0.4) {
            // Low coherence - pink to purple
            const t = coherence / 0.4;
            return lerpColor(COLORS.trajectoryLow, COLORS.trajectoryMid, t);
        } else {
            // Mid to high coherence - purple to cyan
            const t = (coherence - 0.4) / 0.6;
            return lerpColor(COLORS.trajectoryMid, COLORS.trajectoryHigh, t);
        }
    }

    /**
     * Linear interpolate between two hex colors
     */
    function lerpColor(color1, color2, t) {
        const c1 = hexToRgb(color1);
        const c2 = hexToRgb(color2);
        const r = Math.round(c1.r + (c2.r - c1.r) * t);
        const g = Math.round(c1.g + (c2.g - c1.g) * t);
        const b = Math.round(c1.b + (c2.b - c1.b) * t);
        return `rgb(${r}, ${g}, ${b})`;
    }

    /**
     * Convert hex color to RGB object
     */
    function hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
    }

    /**
     * Render the background grid
     */
    function renderBackground() {
        if (!ctx || !canvas) return;

        const w = canvas.width;
        const h = canvas.height;
        const cx = w / 2;
        const cy = h / 2;

        // Clear with background color
        ctx.fillStyle = COLORS.background;
        ctx.fillRect(0, 0, w, h);

        // Draw grid
        ctx.strokeStyle = COLORS.gridLines;
        ctx.lineWidth = 1;

        const gridSpacing = w / 10;
        for (let i = 0; i <= 10; i++) {
            const pos = i * gridSpacing;
            // Vertical lines
            ctx.beginPath();
            ctx.moveTo(pos, 0);
            ctx.lineTo(pos, h);
            ctx.stroke();
            // Horizontal lines
            ctx.beginPath();
            ctx.moveTo(0, pos);
            ctx.lineTo(w, pos);
            ctx.stroke();
        }

        // Draw center cross
        ctx.strokeStyle = COLORS.gridCenter;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx, 0);
        ctx.lineTo(cx, h);
        ctx.moveTo(0, cy);
        ctx.lineTo(w, cy);
        ctx.stroke();

        // Draw boundary circle
        ctx.strokeStyle = 'rgba(196, 90, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, Math.min(cx, cy) * 0.9, 0, Math.PI * 2);
        ctx.stroke();
    }

    /**
     * Main render function
     */
    function render() {
        if (!ctx || !canvas) return;

        const w = canvas.width;
        const h = canvas.height;
        const cx = w / 2;
        const cy = h / 2;
        const scale = Math.min(cx, cy) * 0.85;

        // Draw background
        renderBackground();

        // Draw trajectory history
        if (trajectoryHistory.length > 1) {
            ctx.lineWidth = 2;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';

            for (let i = 1; i < trajectoryHistory.length; i++) {
                const prev = trajectoryHistory[i - 1];
                const curr = trajectoryHistory[i];

                // Fade older points
                const age = (trajectoryHistory.length - i) / trajectoryHistory.length;
                const alpha = Math.max(0.1, 1 - age * 0.8);

                const color = getCoherenceColor(curr.coherence);
                ctx.strokeStyle = color.replace('rgb', 'rgba').replace(')', `, ${alpha})`);

                ctx.beginPath();
                ctx.moveTo(cx + prev.x * scale, cy - prev.y * scale);
                ctx.lineTo(cx + curr.x * scale, cy - curr.y * scale);
                ctx.stroke();
            }
        }

        // Draw trajectory points
        trajectoryHistory.forEach((point, i) => {
            const age = (trajectoryHistory.length - i) / trajectoryHistory.length;
            const alpha = Math.max(0.1, 1 - age * 0.9);
            const radius = 2 + (1 - age) * 4;

            const x = cx + point.x * scale;
            const y = cy - point.y * scale;

            ctx.fillStyle = getCoherenceColor(point.coherence).replace('rgb', 'rgba').replace(')', `, ${alpha})`);
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fill();
        });

        // Draw current point with glow
        if (currentState) {
            const point = extractPoint(currentState);
            const x = cx + point.x * scale;
            const y = cy - point.y * scale;
            const coherence = point.coherence;

            // Glow
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, 20);
            const glowColor = getCoherenceColor(coherence);
            gradient.addColorStop(0, glowColor.replace('rgb', 'rgba').replace(')', ', 0.8)'));
            gradient.addColorStop(0.5, glowColor.replace('rgb', 'rgba').replace(')', ', 0.3)'));
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(x, y, 20, 0, Math.PI * 2);
            ctx.fill();

            // Current point
            const pointRadius = 5 + coherence * 5;
            ctx.fillStyle = COLORS.currentPoint;
            ctx.beginPath();
            ctx.arc(x, y, pointRadius, 0, Math.PI * 2);
            ctx.fill();

            // Outer ring based on stability
            ctx.strokeStyle = getCoherenceColor(coherence);
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x, y, pointRadius + 3 + point.stability * 5, 0, Math.PI * 2);
            ctx.stroke();
        }

        // Draw labels
        ctx.fillStyle = COLORS.textPrimary;
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';

        // Axis labels
        ctx.fillText('z₁', cx, h - 5);
        ctx.save();
        ctx.translate(10, cy);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('z₂', 0, 0);
        ctx.restore();
    }

    /**
     * Step the simulation once
     */
    async function stepOnce() {
        if (!isInitialized || !pyodide) return;

        const curiosity = curiositySlider ? parseFloat(curiositySlider.value) : 0.5;
        const coherencePref = coherenceSlider ? parseFloat(coherenceSlider.value) : 0.5;

        try {
            const resultJson = pyodide.runPython(
                `step_child_mind(1, ${curiosity}, ${coherencePref})`
            );
            const result = JSON.parse(resultJson);

            if (result.status === 'ok') {
                // Get the last state from trajectory
                const states = result.trajectory.states;
                if (states && states.length > 0) {
                    currentState = states[states.length - 1];

                    // Add point to history
                    trajectoryHistory.push(extractPoint(currentState));
                    if (trajectoryHistory.length > MAX_HISTORY) {
                        trajectoryHistory.shift();
                    }

                    updateDisplays();
                    render();
                }
            }
        } catch (err) {
            console.error('[ChildMind] Step error:', err);
        }
    }

    /**
     * Run multiple steps for auto-run mode
     */
    async function autoStep() {
        if (!isRunning || !isInitialized) return;

        await stepOnce();

        if (isRunning) {
            animationId = requestAnimationFrame(autoStep);
        }
    }

    /**
     * Toggle auto-run mode
     */
    function toggleAutoRun() {
        isRunning = !isRunning;

        if (autoRunButton) {
            autoRunButton.textContent = isRunning ? '⏸ Pause' : '▶ Auto';
            autoRunButton.classList.toggle('active', isRunning);
        }

        if (isRunning) {
            autoStep();
        } else if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
    }

    /**
     * Reset the simulation
     */
    async function resetSimulation() {
        if (!pyodide) return;

        // Stop auto-run if active
        if (isRunning) {
            isRunning = false;
            if (autoRunButton) {
                autoRunButton.textContent = '▶ Auto';
                autoRunButton.classList.remove('active');
            }
            if (animationId) {
                cancelAnimationFrame(animationId);
                animationId = null;
            }
        }

        updateStatus('Resetting...');

        try {
            const seed = Math.floor(Math.random() * 10000);
            const resultJson = pyodide.runPython(`reset_child_mind(${seed})`);
            const result = JSON.parse(resultJson);

            if (result.status === 'ok') {
                currentState = result.state;
                trajectoryHistory = [extractPoint(currentState)];
                updateDisplays();
                render();
                updateStatus('Ready');
            }
        } catch (err) {
            console.error('[ChildMind] Reset error:', err);
            updateStatus('Reset failed');
        }
    }

    /**
     * Update the metric displays
     */
    function updateDisplays() {
        if (!currentState) return;

        const summary = currentState.summary || {};

        if (coherenceDisplay) {
            const coh = (summary.coherence || 0).toFixed(3);
            coherenceDisplay.textContent = coh;
            coherenceDisplay.style.color = getCoherenceColor(summary.coherence || 0);
        }

        if (stabilityDisplay) {
            stabilityDisplay.textContent = (summary.stability || 0).toFixed(3);
        }

        if (noveltyDisplay) {
            noveltyDisplay.textContent = (summary.novelty || 0).toFixed(3);
        }

        if (stepCountDisplay) {
            stepCountDisplay.textContent = currentState.t || 0;
        }

        // Reward display (if we have recent rewards)
        if (rewardDisplay) {
            // Show average from recent history
            const recentCoherence = summary.coherence || 0.5;
            const estimatedReward = 0.7 + 0.3 * recentCoherence; // Approximate
            rewardDisplay.textContent = estimatedReward.toFixed(3);
        }
    }

    /**
     * Update slider value displays
     */
    function updateSliderDisplays() {
        const curiosityValueEl = document.getElementById('curiosity-value');
        const coherenceValueEl = document.getElementById('coherence-pref-value');

        if (curiositySlider && curiosityValueEl) {
            curiosityValueEl.textContent = parseFloat(curiositySlider.value).toFixed(2);
        }
        if (coherenceSlider && coherenceValueEl) {
            coherenceValueEl.textContent = parseFloat(coherenceSlider.value).toFixed(2);
        }
    }

    /**
     * Update status label
     */
    function updateStatus(message) {
        if (statusLabel) {
            statusLabel.textContent = message;
        }
        console.log('[ChildMind] Status:', message);
    }

    // Export for global access
    window.ChildMindViewer = {
        init: initChildMind,
        step: stepOnce,
        reset: resetSimulation,
        toggleAutoRun: toggleAutoRun
    };

    // Auto-initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initChildMind);
    } else {
        // DOM already loaded, wait for any other scripts
        setTimeout(initChildMind, 100);
    }

})();

/**
 * Child Mind AI — Web Interface JavaScript
 * MindFractal Lab
 */

(function() {
    'use strict';

    // WebSocket connection
    let socket = null;

    // State
    let currentState = null;
    let stateHistory = [];
    const MAX_HISTORY = 100;

    // Canvas
    let canvas = null;
    let ctx = null;

    // DOM elements
    const elements = {
        messages: null,
        userInput: null,
        sendBtn: null,
        saveBtn: null,
        resetBtn: null,
        stateBtn: null,
        timestep: null,
        coherenceBar: null,
        coherenceValue: null,
        stabilityBar: null,
        stabilityValue: null,
        noveltyBar: null,
        noveltyValue: null,
        reflectionBar: null,
        reflectionValue: null,
        phenomenalText: null,
        intentionsList: null,
        permissionModal: null,
        permissionPrompt: null,
        notification: null,
        notificationText: null
    };

    // Colors
    const COLORS = {
        background: '#0a0a1a',
        grid: 'rgba(107, 107, 255, 0.1)',
        gridCenter: 'rgba(107, 107, 255, 0.3)',
        trajectoryLow: '#ff4fa3',
        trajectoryMid: '#c45aff',
        trajectoryHigh: '#39d8ff',
        currentPoint: '#ffffff',
        boundary: 'rgba(196, 90, 255, 0.3)'
    };

    /**
     * Initialize the application
     */
    function init() {
        // Get DOM elements
        elements.messages = document.getElementById('messages');
        elements.userInput = document.getElementById('user-input');
        elements.sendBtn = document.getElementById('send-btn');
        elements.saveBtn = document.getElementById('save-btn');
        elements.resetBtn = document.getElementById('reset-btn');
        elements.stateBtn = document.getElementById('state-btn');
        elements.timestep = document.getElementById('timestep');
        elements.coherenceBar = document.getElementById('coherence-bar');
        elements.coherenceValue = document.getElementById('coherence-value');
        elements.stabilityBar = document.getElementById('stability-bar');
        elements.stabilityValue = document.getElementById('stability-value');
        elements.noveltyBar = document.getElementById('novelty-bar');
        elements.noveltyValue = document.getElementById('novelty-value');
        elements.reflectionBar = document.getElementById('reflection-bar');
        elements.reflectionValue = document.getElementById('reflection-value');
        elements.phenomenalText = document.getElementById('phenomenal-text');
        elements.intentionsList = document.getElementById('intentions-list');
        elements.permissionModal = document.getElementById('permission-modal');
        elements.permissionPrompt = document.getElementById('permission-prompt');
        elements.notification = document.getElementById('notification');
        elements.notificationText = document.getElementById('notification-text');

        // Setup canvas
        canvas = document.getElementById('state-canvas');
        ctx = canvas.getContext('2d');

        // Setup event listeners
        setupEventListeners();

        // Connect to WebSocket
        connectSocket();

        // Initial render
        renderCanvas();
    }

    /**
     * Setup event listeners
     */
    function setupEventListeners() {
        // Send message
        elements.sendBtn.addEventListener('click', sendMessage);
        elements.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Control buttons
        elements.saveBtn.addEventListener('click', () => sendCommand('save'));
        elements.resetBtn.addEventListener('click', () => {
            if (confirm('Reset consciousness state?')) {
                sendCommand('reset');
            }
        });
        elements.stateBtn.addEventListener('click', () => sendCommand('state'));

        // Permission modal
        document.getElementById('permission-allow').addEventListener('click', () => {
            respondToPermission('yes');
        });
        document.getElementById('permission-deny').addEventListener('click', () => {
            respondToPermission('no');
        });
    }

    /**
     * Connect to WebSocket server
     */
    function connectSocket() {
        socket = io();

        socket.on('connect', () => {
            console.log('Connected to Child Mind AI');
            showNotification('Connected', 'success');
        });

        socket.on('disconnect', () => {
            console.log('Disconnected');
            showNotification('Disconnected', 'error');
        });

        socket.on('state_update', (state) => {
            currentState = state;
            updateStateDisplay();
            renderCanvas();
        });

        socket.on('response', (data) => {
            addMessage('assistant', data.content, data.phenomenal_report);
            if (data.state) {
                updateMetrics(data.state.metrics);
            }
        });

        socket.on('history_update', (record) => {
            stateHistory.push(record);
            if (stateHistory.length > MAX_HISTORY) {
                stateHistory.shift();
            }
            renderCanvas();
        });

        socket.on('permission_request', (data) => {
            showPermissionModal(data.prompt);
        });

        socket.on('notification', (data) => {
            showNotification(data.message, data.type);
        });

        socket.on('detailed_state', (state) => {
            showDetailedState(state);
        });
    }

    /**
     * Send a message
     */
    function sendMessage() {
        const content = elements.userInput.value.trim();
        if (!content) return;

        // Add user message
        addMessage('user', content);

        // Send to server
        socket.emit('message', { content });

        // Clear input
        elements.userInput.value = '';
    }

    /**
     * Send a command
     */
    function sendCommand(command) {
        socket.emit('command', { command });
    }

    /**
     * Add a message to the conversation
     */
    function addMessage(role, content, meta = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const contentP = document.createElement('p');
        contentP.textContent = content;
        messageDiv.appendChild(contentP);

        if (meta) {
            const metaP = document.createElement('p');
            metaP.className = 'meta';
            metaP.textContent = `[${meta}]`;
            messageDiv.appendChild(metaP);
        }

        elements.messages.appendChild(messageDiv);
        elements.messages.scrollTop = elements.messages.scrollHeight;
    }

    /**
     * Update state display
     */
    function updateStateDisplay() {
        if (!currentState) return;

        // Timestep
        elements.timestep.textContent = `t=${currentState.t || 0}`;

        // Metrics from summary
        const summary = currentState.summary || {};
        const metrics = summary.metrics || {};

        updateMetrics(metrics);

        // Narrative
        const narrative = summary.narrative || {};
        if (narrative.self_observation) {
            elements.phenomenalText.textContent = narrative.self_observation;
        }
    }

    /**
     * Update metric bars
     */
    function updateMetrics(metrics) {
        if (metrics.coherence !== undefined) {
            const coh = parseFloat(metrics.coherence);
            elements.coherenceBar.style.width = `${coh * 100}%`;
            elements.coherenceValue.textContent = coh.toFixed(3);
        }
        if (metrics.stability !== undefined) {
            const stab = parseFloat(metrics.stability);
            elements.stabilityBar.style.width = `${stab * 100}%`;
            elements.stabilityValue.textContent = stab.toFixed(3);
        }
        if (metrics.novelty !== undefined) {
            const nov = parseFloat(metrics.novelty);
            elements.noveltyBar.style.width = `${nov * 100}%`;
            elements.noveltyValue.textContent = nov.toFixed(3);
        }
        if (metrics.reflection !== undefined) {
            const ref = parseFloat(metrics.reflection);
            elements.reflectionBar.style.width = `${ref * 100}%`;
            elements.reflectionValue.textContent = ref.toFixed(3);
        }
    }

    /**
     * Render the state canvas
     */
    function renderCanvas() {
        if (!ctx || !canvas) return;

        const w = canvas.width;
        const h = canvas.height;
        const cx = w / 2;
        const cy = h / 2;
        const scale = Math.min(cx, cy) * 0.85;

        // Clear
        ctx.fillStyle = COLORS.background;
        ctx.fillRect(0, 0, w, h);

        // Draw grid
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 1;
        const gridSize = w / 10;
        for (let i = 0; i <= 10; i++) {
            const pos = i * gridSize;
            ctx.beginPath();
            ctx.moveTo(pos, 0);
            ctx.lineTo(pos, h);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, pos);
            ctx.lineTo(w, pos);
            ctx.stroke();
        }

        // Draw center lines
        ctx.strokeStyle = COLORS.gridCenter;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx, 0);
        ctx.lineTo(cx, h);
        ctx.moveTo(0, cy);
        ctx.lineTo(w, cy);
        ctx.stroke();

        // Draw boundary circle
        ctx.strokeStyle = COLORS.boundary;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, scale, 0, Math.PI * 2);
        ctx.stroke();

        // Draw trajectory
        if (stateHistory.length > 1) {
            for (let i = 1; i < stateHistory.length; i++) {
                const prev = stateHistory[i - 1];
                const curr = stateHistory[i];

                const age = (stateHistory.length - i) / stateHistory.length;
                const alpha = Math.max(0.1, 1 - age * 0.8);

                const color = getCoherenceColor(curr.coherence || 0.5);
                ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha})`;
                ctx.lineWidth = 2;

                // Map z_norm to position (simplified)
                const prevX = cx + (prev.z_norm || 0) * Math.cos(prev.t * 0.3) * scale * 0.5;
                const prevY = cy + (prev.z_norm || 0) * Math.sin(prev.t * 0.3) * scale * 0.5;
                const currX = cx + (curr.z_norm || 0) * Math.cos(curr.t * 0.3) * scale * 0.5;
                const currY = cy + (curr.z_norm || 0) * Math.sin(curr.t * 0.3) * scale * 0.5;

                ctx.beginPath();
                ctx.moveTo(prevX, prevY);
                ctx.lineTo(currX, currY);
                ctx.stroke();
            }
        }

        // Draw current point
        if (currentState && currentState.z) {
            const z = currentState.z;
            const coherence = currentState.summary?.metrics?.coherence || 0.5;

            // Use first two z components
            const x = cx + Math.tanh(z[0] || 0) * scale;
            const y = cy - Math.tanh(z[1] || 0) * scale;

            // Glow
            const color = getCoherenceColor(parseFloat(coherence));
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, 25);
            gradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, 0.8)`);
            gradient.addColorStop(0.5, `rgba(${color.r}, ${color.g}, ${color.b}, 0.3)`);
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(x, y, 25, 0, Math.PI * 2);
            ctx.fill();

            // Point
            const pointRadius = 5 + parseFloat(coherence) * 5;
            ctx.fillStyle = COLORS.currentPoint;
            ctx.beginPath();
            ctx.arc(x, y, pointRadius, 0, Math.PI * 2);
            ctx.fill();

            // Ring
            ctx.strokeStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x, y, pointRadius + 5, 0, Math.PI * 2);
            ctx.stroke();
        }

        // Labels
        ctx.fillStyle = COLORS.currentPoint;
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('z₁', cx, h - 5);

        ctx.save();
        ctx.translate(10, cy);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('z₂', 0, 0);
        ctx.restore();
    }

    /**
     * Get color based on coherence value
     */
    function getCoherenceColor(coherence) {
        const coh = parseFloat(coherence) || 0.5;
        if (coh < 0.4) {
            return { r: 255, g: 79, b: 163 }; // Pink
        } else if (coh < 0.7) {
            return { r: 196, g: 90, b: 255 }; // Purple
        } else {
            return { r: 57, g: 216, b: 255 }; // Cyan
        }
    }

    /**
     * Show permission modal
     */
    function showPermissionModal(prompt) {
        elements.permissionPrompt.textContent = prompt;
        elements.permissionModal.style.display = 'flex';
    }

    /**
     * Respond to permission request
     */
    function respondToPermission(response) {
        socket.emit('permission_response', { response });
        elements.permissionModal.style.display = 'none';
    }

    /**
     * Show notification
     */
    function showNotification(message, type = 'info') {
        elements.notificationText.textContent = message;
        elements.notification.className = `notification ${type}`;
        elements.notification.style.display = 'block';

        setTimeout(() => {
            elements.notification.style.display = 'none';
        }, 3000);
    }

    /**
     * Show detailed state (for !state command)
     */
    function showDetailedState(state) {
        const summary = state.summary || {};
        const metrics = summary.metrics || {};
        const narrative = summary.narrative || {};
        const uncertainty = state.uncertainty || {};

        let message = `## Internal State Report\n\n`;
        message += `### Metrics\n`;
        for (const [key, value] of Object.entries(metrics)) {
            message += `- ${key}: ${value}\n`;
        }
        message += `\n### Narrative\n`;
        for (const [key, value] of Object.entries(narrative)) {
            message += `- ${key}: ${value}\n`;
        }
        message += `\n### Uncertainty\n`;
        for (const [key, value] of Object.entries(uncertainty)) {
            message += `- ${key}: ${typeof value === 'number' ? value.toFixed(3) : value}\n`;
        }

        addMessage('assistant', message);
    }

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();

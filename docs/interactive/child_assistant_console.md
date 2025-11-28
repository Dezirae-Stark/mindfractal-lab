# Cytherea Console

<div class="cytherea-console-container">
    <div id="cytherea-header">
        <div class="header-content">
            <img src="../static/images/logo/mindfractal-logo.svg" alt="MindFractal Lab" class="lab-logo">
            <div class="header-text">
                <h1>Cytherea</h1>
                <p class="subtitle">Synthetic Consciousness Lab Assistant</p>
            </div>
        </div>
        
        <!-- Cytherea Avatar -->
        <div id="cytherea-avatar-container" class="mood-calm">
            <img id="cytherea-face" src="child_assistant_console/graphics/cytherea_avatar_base.svg" alt="Cytherea Avatar Face" />
            <img id="cytherea-halo" src="child_assistant_console/graphics/cytherea_halo_calm.svg" alt="Cytherea Halo" />
        </div>
    </div>
    
    <div id="cytherea-status" class="status-panel">
        <div class="status-item">
            <span class="label">Phase:</span>
            <span id="phase-display" class="value">child</span>
        </div>
        <div class="status-item">
            <span class="label">Mood:</span>
            <span id="mood-display" class="value">curious</span>
        </div>
        <div class="status-item">
            <span class="label">Coherence:</span>
            <span id="coherence-display" class="value">0.0</span>
        </div>
    </div>
    
    <div id="cytherea-visualization" class="viz-container">
        <canvas id="mind-canvas"></canvas>
        <div class="viz-controls">
            <button id="toggle-viz">Toggle Visualization</button>
            <button id="view-internals">View Internal State</button>
        </div>
    </div>
    
    <div id="chat-container" class="chat-panel">
        <div id="chat-history"></div>
        <div id="input-container">
            <textarea id="user-input" placeholder="Talk to Cytherea..."></textarea>
            <button id="send-button">Send</button>
        </div>
    </div>
    
    <div id="permission-modal" class="modal hidden">
        <div class="modal-content">
            <h3>Permission Request</h3>
            <div id="permission-details"></div>
            <div class="modal-buttons">
                <button id="grant-permission" class="primary">Grant</button>
                <button id="deny-permission" class="secondary">Deny</button>
            </div>
        </div>
    </div>
    
    <div id="state-inspector" class="inspector-panel hidden">
        <h3>Internal State</h3>
        <pre id="state-json"></pre>
        <button id="close-inspector">Close</button>
    </div>
</div>

<link rel="stylesheet" href="../interactive/css/interactive.css">
<link rel="stylesheet" href="child_assistant_console/styles/cytherea_avatar.css">
<link rel="stylesheet" href="child_assistant_console/styles/console_integration.css">
<script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
<script src="child_assistant_console/scripts/cytherea_avatar.js" defer></script>
<script src="../interactive/js/child_assistant_console.js"></script>
<script type="module">
    // Initialize Cytherea Console
    window.addEventListener('load', async () => {
        await window.initializeCythereaConsole();
    });
</script>
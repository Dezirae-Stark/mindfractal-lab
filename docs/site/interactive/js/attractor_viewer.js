/**
 * Attractor Viewer — 3D Attractor Explorer
 * MindFractal Lab
 *
 * Interactive 3D attractor visualization with orbit rendering and analysis.
 */

class AttractorViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container #${containerId} not found`);
            return;
        }

        // Parameters
        this.params = {
            c1: 0.1,
            c2: 0.1,
            c3: 0.0,
            x0: [0.1, 0.1, 0.1],
            n_steps: 2000,
            elev: 30,
            azim: 45
        };

        this.init();
    }

    init() {
        this.createUI();
        this.bindEvents();
    }

    createUI() {
        this.container.innerHTML = `
            <div class="attractor-viewer interactive-demo">
                <div class="viewer-header">
                    <h3>3D Attractor Explorer</h3>
                    <p class="explorer-only">Explore strange attractors in 3D parameter space</p>
                </div>

                <div class="viewer-layout">
                    <div class="controls-panel">
                        <div class="control-section">
                            <h4>Parameters (c)</h4>
                            <div class="control-group">
                                <label>c₁: <span id="c1-val">${this.params.c1}</span></label>
                                <input type="range" id="c1-slider" min="-1.5" max="1.5" step="0.05" value="${this.params.c1}">
                            </div>
                            <div class="control-group">
                                <label>c₂: <span id="c2-val">${this.params.c2}</span></label>
                                <input type="range" id="c2-slider" min="-1.5" max="1.5" step="0.05" value="${this.params.c2}">
                            </div>
                            <div class="control-group">
                                <label>c₃: <span id="c3-val">${this.params.c3}</span></label>
                                <input type="range" id="c3-slider" min="-1.5" max="1.5" step="0.05" value="${this.params.c3}">
                            </div>
                        </div>

                        <div class="control-section">
                            <h4>View Angle</h4>
                            <div class="control-group">
                                <label>Elevation: <span id="elev-val">${this.params.elev}</span>°</label>
                                <input type="range" id="elev-slider" min="0" max="90" step="5" value="${this.params.elev}">
                            </div>
                            <div class="control-group">
                                <label>Azimuth: <span id="azim-val">${this.params.azim}</span>°</label>
                                <input type="range" id="azim-slider" min="0" max="360" step="5" value="${this.params.azim}">
                            </div>
                        </div>

                        <div class="control-section">
                            <h4>Simulation</h4>
                            <div class="control-group">
                                <label>Steps: <span id="steps-val">${this.params.n_steps}</span></label>
                                <input type="range" id="steps-slider" min="500" max="5000" step="500" value="${this.params.n_steps}">
                            </div>
                        </div>

                        <div class="button-row">
                            <button id="compute-attractor-btn" class="primary-btn">Compute Attractor</button>
                            <button id="compute-poincare-btn" class="secondary-btn">Poincare Section</button>
                        </div>

                        <div class="button-row">
                            <button id="scan-types-btn" class="secondary-btn">Scan Types</button>
                        </div>
                    </div>

                    <div class="display-panel">
                        <div id="pyodide-status" class="status-bar">Initializing...</div>
                        <div class="image-container">
                            <img id="attractor-image" class="result-image" alt="3D Attractor">
                        </div>
                        <div id="attractor-info" class="info-panel"></div>
                    </div>
                </div>

                <div class="researcher-only code-panel">
                    <h4>Python Code</h4>
                    <pre><code id="attractor-code">from attractor_core import compute_orbit_3d, render_attractor_3d_to_base64
import numpy as np

x0 = np.array([0.1, 0.1, 0.1])
c = np.array([${this.params.c1}, ${this.params.c2}, ${this.params.c3}])

# Render 3D attractor
image_b64 = render_attractor_3d_to_base64(x0, c, n_steps=${this.params.n_steps})</code></pre>
                </div>
            </div>
        `;

        this.imageEl = document.getElementById('attractor-image');
        this.infoEl = document.getElementById('attractor-info');
        this.statusEl = document.getElementById('pyodide-status');
        this.codeEl = document.getElementById('attractor-code');
    }

    bindEvents() {
        // Parameter sliders
        const sliders = [
            { id: 'c1-slider', param: 'c1', display: 'c1-val' },
            { id: 'c2-slider', param: 'c2', display: 'c2-val' },
            { id: 'c3-slider', param: 'c3', display: 'c3-val' },
            { id: 'elev-slider', param: 'elev', display: 'elev-val' },
            { id: 'azim-slider', param: 'azim', display: 'azim-val' },
            { id: 'steps-slider', param: 'n_steps', display: 'steps-val' }
        ];

        sliders.forEach(({ id, param, display }) => {
            const slider = document.getElementById(id);
            slider.addEventListener('input', (e) => {
                this.params[param] = parseFloat(e.target.value);
                document.getElementById(display).textContent =
                    param === 'n_steps' ? this.params[param] :
                    param.includes('elev') || param.includes('azim') ? this.params[param] :
                    this.params[param].toFixed(2);
                this.updateCode();
            });
        });

        // Buttons
        document.getElementById('compute-attractor-btn').addEventListener('click', () => this.computeAttractor());
        document.getElementById('compute-poincare-btn').addEventListener('click', () => this.computePoincare());
        document.getElementById('scan-types-btn').addEventListener('click', () => this.scanTypes());

        // Auto-compute on ready
        MindFractal.onReady(() => this.computeAttractor());
    }

    updateCode() {
        this.codeEl.textContent = `from attractor_core import compute_orbit_3d, render_attractor_3d_to_base64
import numpy as np

x0 = np.array([0.1, 0.1, 0.1])
c = np.array([${this.params.c1.toFixed(2)}, ${this.params.c2.toFixed(2)}, ${this.params.c3.toFixed(2)}])

# Render 3D attractor
image_b64 = render_attractor_3d_to_base64(x0, c, n_steps=${this.params.n_steps})`;
    }

    async computeAttractor() {
        this.statusEl.textContent = 'Computing 3D attractor...';
        this.imageEl.style.opacity = 0.5;

        try {
            const code = `
from attractor_core import render_attractor_3d_to_base64, classify_attractor_3d, compute_lyapunov_3d
import numpy as np

x0 = np.array([0.1, 0.1, 0.1])
c = np.array([${this.params.c1}, ${this.params.c2}, ${this.params.c3}])

# Classify
atype = classify_attractor_3d(x0, c)
spectrum = compute_lyapunov_3d(x0, c, n_steps=500)

# Render
img = render_attractor_3d_to_base64(x0, c, n_steps=${this.params.n_steps}, elev=${this.params.elev}, azim=${this.params.azim})
(img, atype, spectrum[0], spectrum[1], spectrum[2])
`;
            const result = await MindFractal.runPython(code);
            const [img, atype, l1, l2, l3] = result.toJs();

            this.imageEl.src = 'data:image/png;base64,' + img;
            this.imageEl.style.opacity = 1;

            this.infoEl.innerHTML = `
                <div class="info-row"><strong>Attractor Type:</strong> ${atype}</div>
                <div class="info-row"><strong>Lyapunov Spectrum:</strong> [${formatNumber(l1)}, ${formatNumber(l2)}, ${formatNumber(l3)}]</div>
                <div class="info-row"><strong>Parameters:</strong> c = (${this.params.c1.toFixed(2)}, ${this.params.c2.toFixed(2)}, ${this.params.c3.toFixed(2)})</div>
            `;

            this.statusEl.textContent = 'Done';
        } catch (error) {
            console.error('Computation error:', error);
            this.statusEl.textContent = 'Error: ' + error.message;
        }
    }

    async computePoincare() {
        this.statusEl.textContent = 'Computing Poincare section...';
        this.imageEl.style.opacity = 0.5;

        try {
            const code = `
from attractor_core import render_poincare_to_base64
import numpy as np

x0 = np.array([0.1, 0.1, 0.1])
c = np.array([${this.params.c1}, ${this.params.c2}, ${this.params.c3}])

img = render_poincare_to_base64(x0, c, n_steps=10000, plane_coord=2, plane_value=0.0)
img
`;
            const img = await MindFractal.runPython(code);

            this.imageEl.src = 'data:image/png;base64,' + img;
            this.imageEl.style.opacity = 1;

            this.infoEl.innerHTML = `
                <div class="info-row"><strong>View:</strong> Poincare Section (z = 0 plane)</div>
                <div class="info-row"><strong>Parameters:</strong> c = (${this.params.c1.toFixed(2)}, ${this.params.c2.toFixed(2)}, ${this.params.c3.toFixed(2)})</div>
            `;

            this.statusEl.textContent = 'Done';
        } catch (error) {
            console.error('Computation error:', error);
            this.statusEl.textContent = 'Error: ' + error.message;
        }
    }

    async scanTypes() {
        this.statusEl.textContent = 'Scanning attractor types (this may take a moment)...';
        this.imageEl.style.opacity = 0.5;

        try {
            const code = `
from attractor_core import render_attractor_scan_to_base64

img = render_attractor_scan_to_base64(resolution=25, c1_range=(-1.5, 1.5), c2_range=(-1.5, 1.5))
img
`;
            const img = await MindFractal.runPython(code);

            this.imageEl.src = 'data:image/png;base64,' + img;
            this.imageEl.style.opacity = 1;

            this.infoEl.innerHTML = `
                <div class="info-row"><strong>View:</strong> Attractor Type Map</div>
                <div class="info-row">
                    <span style="color:#3498db;">Fixed Point</span> |
                    <span style="color:#2ecc71;">Limit Cycle</span> |
                    <span style="color:#f1c40f;">Torus</span> |
                    <span style="color:#e74c3c;">Strange</span>
                </div>
            `;

            this.statusEl.textContent = 'Done';
        } catch (error) {
            console.error('Computation error:', error);
            this.statusEl.textContent = 'Error: ' + error.message;
        }
    }
}

// Inject styles
const attractorViewerStyles = `
<style>
.attractor-viewer {
    background: linear-gradient(135deg, rgba(26, 26, 46, 0.95), rgba(22, 33, 62, 0.95));
    border-radius: 16px;
    padding: 24px;
    margin: 20px 0;
}

.attractor-viewer .viewer-header h3 {
    color: #ffffff;
    margin: 0 0 8px 0;
}

.attractor-viewer .viewer-header p {
    color: #a0a0a0;
    margin: 0 0 20px 0;
}

.attractor-viewer .viewer-layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 24px;
}

@media (max-width: 768px) {
    .attractor-viewer .viewer-layout {
        grid-template-columns: 1fr;
    }
}

.attractor-viewer .control-section {
    margin-bottom: 20px;
}

.attractor-viewer .control-section h4 {
    color: #06b6d4;
    margin: 0 0 12px 0;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.attractor-viewer .control-group {
    margin-bottom: 12px;
}

.attractor-viewer .control-group label {
    display: block;
    color: #e0e0e0;
    font-size: 13px;
    margin-bottom: 4px;
}

.attractor-viewer input[type="range"] {
    width: 100%;
    height: 6px;
    background: #2d2250;
    border-radius: 3px;
    outline: none;
    -webkit-appearance: none;
}

.attractor-viewer input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    border-radius: 50%;
    cursor: pointer;
}

.attractor-viewer .button-row {
    display: flex;
    gap: 10px;
    margin-bottom: 10px;
}

.attractor-viewer .primary-btn,
.attractor-viewer .secondary-btn {
    flex: 1;
    padding: 10px 16px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    transition: transform 0.2s, box-shadow 0.2s;
}

.attractor-viewer .primary-btn {
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    color: white;
}

.attractor-viewer .secondary-btn {
    background: rgba(124, 58, 237, 0.2);
    color: #a78bfa;
    border: 1px solid #7c3aed;
}

.attractor-viewer .primary-btn:hover,
.attractor-viewer .secondary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
}

.attractor-viewer .display-panel {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px;
    padding: 16px;
}

.attractor-viewer .status-bar {
    color: #06b6d4;
    font-size: 13px;
    margin-bottom: 12px;
    padding: 8px;
    background: rgba(6, 182, 212, 0.1);
    border-radius: 6px;
}

.attractor-viewer .image-container {
    text-align: center;
    margin-bottom: 16px;
}

.attractor-viewer .result-image {
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    transition: opacity 0.3s;
}

.attractor-viewer .info-panel {
    color: #c0c0c0;
    font-size: 13px;
}

.attractor-viewer .info-row {
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.attractor-viewer .code-panel {
    margin-top: 20px;
    padding: 16px;
    background: rgba(0, 0, 0, 0.4);
    border-radius: 8px;
}

.attractor-viewer .code-panel h4 {
    color: #06b6d4;
    margin: 0 0 12px 0;
}

.attractor-viewer .code-panel pre {
    margin: 0;
    padding: 12px;
    background: #0d1117;
    border-radius: 6px;
    overflow-x: auto;
}

.attractor-viewer .code-panel code {
    color: #c9d1d9;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
</style>
`;

document.head.insertAdjacentHTML('beforeend', attractorViewerStyles);

// Export
window.AttractorViewer = AttractorViewer;

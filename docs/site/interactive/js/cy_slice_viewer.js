/**
 * CY Slice Viewer — Calabi-Yau Dynamics Explorer
 * MindFractal Lab
 *
 * Interactive visualization of CY-inspired complex dynamics.
 */

class CYSliceViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container #${containerId} not found`);
            return;
        }

        // Parameters
        this.params = {
            mode: 'mandelbrot',  // 'mandelbrot', 'julia', 'cy_slice'
            resolution: 200,
            c_re: -0.7,
            c_im: 0.27,
            k: 2,
            eps: 0.5,
            max_iter: 100
        };

        this.init();
    }

    init() {
        this.createUI();
        this.bindEvents();
    }

    createUI() {
        this.container.innerHTML = `
            <div class="cy-viewer interactive-demo">
                <div class="viewer-header">
                    <h3>CY Dynamics Slice Viewer</h3>
                    <p class="explorer-only">Explore Calabi-Yau inspired complex dynamics and fractals</p>
                </div>

                <div class="viewer-layout">
                    <div class="controls-panel">
                        <div class="control-section">
                            <h4>Visualization Mode</h4>
                            <div class="mode-buttons">
                                <button id="mode-mandelbrot" class="mode-btn active">Mandelbrot</button>
                                <button id="mode-julia" class="mode-btn">Julia Set</button>
                                <button id="mode-cy" class="mode-btn">CY Slice</button>
                            </div>
                        </div>

                        <div class="control-section julia-controls" style="display:none;">
                            <h4>Julia Parameter (c)</h4>
                            <div class="control-group">
                                <label>Re(c): <span id="cre-val">${this.params.c_re}</span></label>
                                <input type="range" id="cre-slider" min="-2" max="2" step="0.01" value="${this.params.c_re}">
                            </div>
                            <div class="control-group">
                                <label>Im(c): <span id="cim-val">${this.params.c_im}</span></label>
                                <input type="range" id="cim-slider" min="-2" max="2" step="0.01" value="${this.params.c_im}">
                            </div>
                        </div>

                        <div class="control-section cy-controls" style="display:none;">
                            <h4>CY Parameters</h4>
                            <div class="control-group">
                                <label>k (degree): <span id="k-val">${this.params.k}</span></label>
                                <input type="range" id="k-slider" min="2" max="6" step="1" value="${this.params.k}">
                            </div>
                            <div class="control-group">
                                <label>eps (perturbation): <span id="eps-val">${this.params.eps}</span></label>
                                <input type="range" id="eps-slider" min="0" max="1" step="0.05" value="${this.params.eps}">
                            </div>
                        </div>

                        <div class="control-section">
                            <h4>Rendering</h4>
                            <div class="control-group">
                                <label>Resolution: <span id="res-val">${this.params.resolution}</span></label>
                                <input type="range" id="res-slider" min="100" max="400" step="50" value="${this.params.resolution}">
                            </div>
                            <div class="control-group">
                                <label>Max Iterations: <span id="iter-val">${this.params.max_iter}</span></label>
                                <input type="range" id="iter-slider" min="50" max="300" step="25" value="${this.params.max_iter}">
                            </div>
                        </div>

                        <button id="compute-cy-btn" class="primary-btn">Generate Fractal</button>

                        <div class="presets-section">
                            <h4>Presets</h4>
                            <div class="preset-buttons">
                                <button class="preset-btn" data-preset="dendrite">Dendrite</button>
                                <button class="preset-btn" data-preset="spiral">Spiral</button>
                                <button class="preset-btn" data-preset="rabbit">Rabbit</button>
                                <button class="preset-btn" data-preset="seahorse">Seahorse</button>
                            </div>
                        </div>
                    </div>

                    <div class="display-panel">
                        <div id="cy-status" class="status-bar">Initializing...</div>
                        <div class="image-container">
                            <img id="cy-image" class="result-image" alt="CY Fractal">
                        </div>
                        <div id="cy-info" class="info-panel"></div>
                    </div>
                </div>

                <div class="researcher-only code-panel">
                    <h4>Python Code</h4>
                    <pre><code id="cy-code">from cy_core import compute_and_render_mandelbrot
image_b64 = compute_and_render_mandelbrot(resolution=${this.params.resolution})</code></pre>
                </div>
            </div>
        `;

        this.imageEl = document.getElementById('cy-image');
        this.infoEl = document.getElementById('cy-info');
        this.statusEl = document.getElementById('cy-status');
        this.codeEl = document.getElementById('cy-code');
    }

    bindEvents() {
        // Mode buttons
        ['mandelbrot', 'julia', 'cy'].forEach(mode => {
            document.getElementById(`mode-${mode}`).addEventListener('click', (e) => {
                document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');
                this.params.mode = mode === 'cy' ? 'cy_slice' : mode;
                this.updateControlVisibility();
                this.updateCode();
            });
        });

        // Parameter sliders
        const sliders = [
            { id: 'cre-slider', param: 'c_re', display: 'cre-val' },
            { id: 'cim-slider', param: 'c_im', display: 'cim-val' },
            { id: 'k-slider', param: 'k', display: 'k-val' },
            { id: 'eps-slider', param: 'eps', display: 'eps-val' },
            { id: 'res-slider', param: 'resolution', display: 'res-val' },
            { id: 'iter-slider', param: 'max_iter', display: 'iter-val' }
        ];

        sliders.forEach(({ id, param, display }) => {
            const slider = document.getElementById(id);
            slider.addEventListener('input', (e) => {
                this.params[param] = parseFloat(e.target.value);
                document.getElementById(display).textContent =
                    Number.isInteger(this.params[param]) ? this.params[param] : this.params[param].toFixed(2);
                this.updateCode();
            });
        });

        // Compute button
        document.getElementById('compute-cy-btn').addEventListener('click', () => this.compute());

        // Presets
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => this.applyPreset(btn.dataset.preset));
        });

        // Auto-compute on ready
        MindFractal.onReady(() => this.compute());
    }

    updateControlVisibility() {
        document.querySelector('.julia-controls').style.display =
            this.params.mode === 'julia' ? 'block' : 'none';
        document.querySelector('.cy-controls').style.display =
            this.params.mode === 'cy_slice' ? 'block' : 'none';
    }

    updateCode() {
        let code;
        if (this.params.mode === 'mandelbrot') {
            code = `from cy_core import compute_and_render_mandelbrot
image_b64 = compute_and_render_mandelbrot(resolution=${this.params.resolution})`;
        } else if (this.params.mode === 'julia') {
            code = `from cy_core import compute_and_render_julia
image_b64 = compute_and_render_julia(${this.params.c_re}, ${this.params.c_im}, resolution=${this.params.resolution})`;
        } else {
            code = `from cy_core import compute_and_render_cy_slice
image_b64 = compute_and_render_cy_slice(resolution=${this.params.resolution}, k=${this.params.k}, eps=${this.params.eps})`;
        }
        this.codeEl.textContent = code;
    }

    applyPreset(preset) {
        const presets = {
            dendrite: { c_re: 0, c_im: 1 },
            spiral: { c_re: -0.7463, c_im: 0.1102 },
            rabbit: { c_re: -0.123, c_im: 0.745 },
            seahorse: { c_re: -0.75, c_im: 0.11 }
        };

        if (presets[preset]) {
            this.params.mode = 'julia';
            this.params.c_re = presets[preset].c_re;
            this.params.c_im = presets[preset].c_im;

            // Update UI
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('mode-julia').classList.add('active');
            document.getElementById('cre-slider').value = this.params.c_re;
            document.getElementById('cim-slider').value = this.params.c_im;
            document.getElementById('cre-val').textContent = this.params.c_re;
            document.getElementById('cim-val').textContent = this.params.c_im;

            this.updateControlVisibility();
            this.updateCode();
            this.compute();
        }
    }

    async compute() {
        this.statusEl.textContent = `Computing ${this.params.mode}...`;
        this.imageEl.style.opacity = 0.5;

        try {
            let code;
            let info;

            if (this.params.mode === 'mandelbrot') {
                code = `
from cy_core import compute_and_render_mandelbrot
compute_and_render_mandelbrot(${this.params.resolution})
`;
                info = `<div class="info-row"><strong>Mode:</strong> Mandelbrot Set</div>
                        <div class="info-row"><strong>Resolution:</strong> ${this.params.resolution}x${this.params.resolution}</div>`;
            } else if (this.params.mode === 'julia') {
                code = `
from cy_core import compute_and_render_julia
compute_and_render_julia(${this.params.c_re}, ${this.params.c_im}, ${this.params.resolution})
`;
                info = `<div class="info-row"><strong>Mode:</strong> Julia Set</div>
                        <div class="info-row"><strong>c:</strong> ${this.params.c_re} + ${this.params.c_im}i</div>
                        <div class="info-row"><strong>Resolution:</strong> ${this.params.resolution}x${this.params.resolution}</div>`;
            } else {
                code = `
from cy_core import compute_and_render_cy_slice
compute_and_render_cy_slice(${this.params.resolution}, ${this.params.k}, ${this.params.eps})
`;
                info = `<div class="info-row"><strong>Mode:</strong> CY Slice</div>
                        <div class="info-row"><strong>k:</strong> ${this.params.k}, eps: ${this.params.eps}</div>
                        <div class="info-row"><strong>Resolution:</strong> ${this.params.resolution}x${this.params.resolution}</div>`;
            }

            const img = await MindFractal.runPython(code);

            this.imageEl.src = 'data:image/png;base64,' + img;
            this.imageEl.style.opacity = 1;
            this.infoEl.innerHTML = info;
            this.statusEl.textContent = 'Done';

        } catch (error) {
            console.error('Computation error:', error);
            this.statusEl.textContent = 'Error: ' + error.message;
        }
    }
}

// Inject styles
const cyViewerStyles = `
<style>
.cy-viewer {
    background: linear-gradient(135deg, rgba(26, 26, 46, 0.95), rgba(22, 33, 62, 0.95));
    border-radius: 16px;
    padding: 24px;
    margin: 20px 0;
}

.cy-viewer .viewer-header h3 {
    color: #ffffff;
    margin: 0 0 8px 0;
}

.cy-viewer .viewer-header p {
    color: #a0a0a0;
    margin: 0 0 20px 0;
}

.cy-viewer .viewer-layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 24px;
}

@media (max-width: 768px) {
    .cy-viewer .viewer-layout {
        grid-template-columns: 1fr;
    }
}

.cy-viewer .control-section {
    margin-bottom: 20px;
}

.cy-viewer .control-section h4 {
    color: #06b6d4;
    margin: 0 0 12px 0;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.cy-viewer .mode-buttons {
    display: flex;
    gap: 8px;
}

.cy-viewer .mode-btn {
    flex: 1;
    padding: 8px 12px;
    background: rgba(124, 58, 237, 0.2);
    border: 1px solid #7c3aed;
    color: #a78bfa;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s;
}

.cy-viewer .mode-btn.active {
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    color: white;
    border-color: transparent;
}

.cy-viewer .control-group {
    margin-bottom: 12px;
}

.cy-viewer .control-group label {
    display: block;
    color: #e0e0e0;
    font-size: 13px;
    margin-bottom: 4px;
}

.cy-viewer input[type="range"] {
    width: 100%;
    height: 6px;
    background: #2d2250;
    border-radius: 3px;
    outline: none;
    -webkit-appearance: none;
}

.cy-viewer input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    border-radius: 50%;
    cursor: pointer;
}

.cy-viewer .primary-btn {
    width: 100%;
    padding: 12px;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 20px;
    transition: transform 0.2s, box-shadow 0.2s;
}

.cy-viewer .primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
}

.cy-viewer .presets-section {
    margin-top: 20px;
}

.cy-viewer .preset-buttons {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}

.cy-viewer .preset-btn {
    padding: 8px;
    background: rgba(6, 182, 212, 0.1);
    border: 1px solid #06b6d4;
    color: #06b6d4;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s;
}

.cy-viewer .preset-btn:hover {
    background: rgba(6, 182, 212, 0.2);
}

.cy-viewer .display-panel {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px;
    padding: 16px;
}

.cy-viewer .status-bar {
    color: #06b6d4;
    font-size: 13px;
    margin-bottom: 12px;
    padding: 8px;
    background: rgba(6, 182, 212, 0.1);
    border-radius: 6px;
}

.cy-viewer .image-container {
    text-align: center;
    margin-bottom: 16px;
}

.cy-viewer .result-image {
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    transition: opacity 0.3s;
}

.cy-viewer .info-panel {
    color: #c0c0c0;
    font-size: 13px;
}

.cy-viewer .info-row {
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.cy-viewer .code-panel {
    margin-top: 20px;
    padding: 16px;
    background: rgba(0, 0, 0, 0.4);
    border-radius: 8px;
}

.cy-viewer .code-panel h4 {
    color: #06b6d4;
    margin: 0 0 12px 0;
}

.cy-viewer .code-panel pre {
    margin: 0;
    padding: 12px;
    background: #0d1117;
    border-radius: 6px;
    overflow-x: auto;
}

.cy-viewer .code-panel code {
    color: #c9d1d9;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
</style>
`;

document.head.insertAdjacentHTML('beforeend', cyViewerStyles);

// Export
window.CYSliceViewer = CYSliceViewer;

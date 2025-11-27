/**
 * Fractal Viewer — Interactive 2D/3D Fractal Explorer
 * MindFractal Lab
 *
 * Provides UI controls and rendering for fractal visualization.
 */

class FractalViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container #${containerId} not found`);
            return;
        }

        this.canvas = null;
        this.ctx = null;
        this.imageElement = null;

        // Parameters
        this.params = {
            c1: 0.1,
            c2: 0.1,
            resolution: 100,
            mode: 'basin'  // 'basin', 'lyapunov', 'orbit'
        };

        this.init();
    }

    init() {
        this.createUI();
        this.bindEvents();
    }

    createUI() {
        this.container.innerHTML = `
            <div class="fractal-viewer">
                <div class="controls">
                    <div class="control-group">
                        <label for="c1-slider">c₁: <span id="c1-value">${this.params.c1}</span></label>
                        <input type="range" id="c1-slider" min="-2" max="2" step="0.1" value="${this.params.c1}">
                    </div>
                    <div class="control-group">
                        <label for="c2-slider">c₂: <span id="c2-value">${this.params.c2}</span></label>
                        <input type="range" id="c2-slider" min="-2" max="2" step="0.1" value="${this.params.c2}">
                    </div>
                    <div class="control-group">
                        <label for="resolution-slider">Resolution: <span id="res-value">${this.params.resolution}</span></label>
                        <input type="range" id="resolution-slider" min="50" max="300" step="50" value="${this.params.resolution}">
                    </div>
                    <div class="control-group">
                        <label>Mode:</label>
                        <select id="mode-select">
                            <option value="basin">Basin of Attraction</option>
                            <option value="lyapunov">Lyapunov Map</option>
                        </select>
                    </div>
                    <button id="compute-btn" class="compute-button">Compute</button>
                </div>
                <div class="display">
                    <div id="pyodide-status" class="status">Initializing...</div>
                    <img id="fractal-image" class="fractal-image" alt="Fractal visualization">
                </div>
                <div class="info">
                    <p id="lyap-info"></p>
                </div>
            </div>
        `;

        this.imageElement = document.getElementById('fractal-image');
    }

    bindEvents() {
        // Slider events
        const c1Slider = document.getElementById('c1-slider');
        const c2Slider = document.getElementById('c2-slider');
        const resSlider = document.getElementById('resolution-slider');
        const modeSelect = document.getElementById('mode-select');
        const computeBtn = document.getElementById('compute-btn');

        c1Slider.addEventListener('input', (e) => {
            this.params.c1 = parseFloat(e.target.value);
            document.getElementById('c1-value').textContent = this.params.c1.toFixed(1);
        });

        c2Slider.addEventListener('input', (e) => {
            this.params.c2 = parseFloat(e.target.value);
            document.getElementById('c2-value').textContent = this.params.c2.toFixed(1);
        });

        resSlider.addEventListener('input', (e) => {
            this.params.resolution = parseInt(e.target.value);
            document.getElementById('res-value').textContent = this.params.resolution;
        });

        modeSelect.addEventListener('change', (e) => {
            this.params.mode = e.target.value;
        });

        computeBtn.addEventListener('click', () => this.compute());

        // Auto-compute on initialization
        MindFractal.onReady(() => {
            this.compute();
        });
    }

    async compute() {
        const statusEl = document.getElementById('pyodide-status');
        const infoEl = document.getElementById('lyap-info');

        try {
            statusEl.textContent = 'Computing...';
            this.imageElement.style.opacity = 0.5;

            let base64Image;

            if (this.params.mode === 'basin') {
                base64Image = await MindFractal.computeBasin(
                    this.params.c1,
                    this.params.c2,
                    this.params.resolution
                );
                infoEl.textContent = `Basin of attraction for c = (${this.params.c1}, ${this.params.c2})`;
            } else if (this.params.mode === 'lyapunov') {
                base64Image = await MindFractal.computeLyapunovMap(
                    this.params.resolution
                );
                infoEl.textContent = 'Lyapunov exponent map: Blue = stable, Red = chaotic';
            }

            this.imageElement.src = 'data:image/png;base64,' + base64Image;
            this.imageElement.style.opacity = 1;
            statusEl.textContent = 'Done';

        } catch (error) {
            console.error('Computation error:', error);
            statusEl.textContent = 'Error: ' + error.message;
        }
    }
}

// CSS for the viewer
const fractalViewerStyles = `
<style>
.fractal-viewer {
    background: rgba(20, 15, 40, 0.8);
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
}

.fractal-viewer .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 20px;
}

.fractal-viewer .control-group {
    flex: 1;
    min-width: 150px;
}

.fractal-viewer label {
    display: block;
    margin-bottom: 5px;
    color: #e0e0e0;
}

.fractal-viewer input[type="range"] {
    width: 100%;
    background: #2d2250;
    border-radius: 4px;
}

.fractal-viewer select {
    width: 100%;
    padding: 8px;
    background: #2d2250;
    color: #e0e0e0;
    border: 1px solid #7c3aed;
    border-radius: 4px;
}

.fractal-viewer .compute-button {
    background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
    transition: transform 0.2s;
}

.fractal-viewer .compute-button:hover {
    transform: translateY(-2px);
}

.fractal-viewer .display {
    text-align: center;
}

.fractal-viewer .status {
    color: #06b6d4;
    margin-bottom: 10px;
}

.fractal-viewer .fractal-image {
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    transition: opacity 0.3s;
}

.fractal-viewer .info {
    margin-top: 15px;
    color: #a0a0a0;
    font-style: italic;
    text-align: center;
}
</style>
`;

// Inject styles
document.head.insertAdjacentHTML('beforeend', fractalViewerStyles);

// Export
window.FractalViewer = FractalViewer;

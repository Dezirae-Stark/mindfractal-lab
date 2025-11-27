/**
 * Embeddings Viewer — ML Latent Space Explorer
 * MindFractal Lab
 *
 * Interactive exploration of trajectory embeddings and dimensionality reduction.
 */

class EmbeddingsViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container #${containerId} not found`);
            return;
        }

        // Parameters
        this.params = {
            mode: 'pca',  // 'pca', 'tsne', 'param_space'
            n_samples: 100,
            c_range: [-1.5, 1.5]
        };

        this.init();
    }

    init() {
        this.createUI();
        this.bindEvents();
    }

    createUI() {
        this.container.innerHTML = `
            <div class="embeddings-viewer interactive-demo">
                <div class="viewer-header">
                    <h3>Trajectory Embedding Explorer</h3>
                    <p class="explorer-only">Visualize dynamical trajectories in latent space</p>
                </div>

                <div class="viewer-layout">
                    <div class="controls-panel">
                        <div class="control-section">
                            <h4>Visualization Method</h4>
                            <div class="mode-buttons">
                                <button id="embed-mode-pca" class="mode-btn active">PCA</button>
                                <button id="embed-mode-tsne" class="mode-btn">t-SNE</button>
                                <button id="embed-mode-param" class="mode-btn">Param Space</button>
                            </div>
                        </div>

                        <div class="control-section">
                            <h4>Sampling Parameters</h4>
                            <div class="control-group">
                                <label>Number of Samples: <span id="embed-samples-val">${this.params.n_samples}</span></label>
                                <input type="range" id="embed-samples-slider" min="50" max="300" step="25" value="${this.params.n_samples}">
                            </div>
                            <div class="control-group">
                                <label>Parameter Range (c): <span id="embed-range-val">[-${Math.abs(this.params.c_range[0])}, ${this.params.c_range[1]}]</span></label>
                                <input type="range" id="embed-range-slider" min="0.5" max="2.5" step="0.25" value="${this.params.c_range[1]}">
                            </div>
                        </div>

                        <button id="compute-embedding-btn" class="primary-btn">Generate Embedding</button>

                        <div class="info-box">
                            <h4>About Embeddings</h4>
                            <p class="explorer-only">
                                Trajectories are characterized by statistical features (mean, variance, path length, curvature)
                                and projected to 2D for visualization. Similar dynamics cluster together.
                            </p>
                            <div class="researcher-only">
                                <p>Features extracted per trajectory:</p>
                                <ul>
                                    <li>Mean & std per dimension</li>
                                    <li>Range per dimension</li>
                                    <li>Path length (normalized)</li>
                                    <li>Mean curvature</li>
                                </ul>
                            </div>
                        </div>

                        <div class="legend-box">
                            <h4>Classification Legend</h4>
                            <div class="legend-item">
                                <span class="legend-color" style="background:#3498db;"></span>
                                <span>Stable (converging)</span>
                            </div>
                            <div class="legend-item">
                                <span class="legend-color" style="background:#2ecc71;"></span>
                                <span>Periodic (oscillating)</span>
                            </div>
                            <div class="legend-item">
                                <span class="legend-color" style="background:#e74c3c;"></span>
                                <span>Chaotic (sensitive)</span>
                            </div>
                            <div class="legend-item">
                                <span class="legend-color" style="background:#7f8c8d;"></span>
                                <span>Divergent (escaping)</span>
                            </div>
                        </div>
                    </div>

                    <div class="display-panel">
                        <div id="embed-status" class="status-bar">Initializing...</div>
                        <div class="image-container">
                            <img id="embed-image" class="result-image" alt="Embedding Visualization">
                        </div>
                        <div id="embed-info" class="info-panel"></div>
                    </div>
                </div>

                <div class="researcher-only code-panel">
                    <h4>Python Code</h4>
                    <pre><code id="embed-code">from embeddings_core import compute_and_render_embedding
img = compute_and_render_embedding(n_samples=${this.params.n_samples}, method='pca')</code></pre>
                </div>
            </div>
        `;

        this.imageEl = document.getElementById('embed-image');
        this.infoEl = document.getElementById('embed-info');
        this.statusEl = document.getElementById('embed-status');
        this.codeEl = document.getElementById('embed-code');
    }

    bindEvents() {
        // Mode buttons
        ['pca', 'tsne', 'param'].forEach(mode => {
            document.getElementById(`embed-mode-${mode}`).addEventListener('click', (e) => {
                document.querySelectorAll('.embeddings-viewer .mode-btn').forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');
                this.params.mode = mode === 'param' ? 'param_space' : mode;
                this.updateCode();
            });
        });

        // Sliders
        document.getElementById('embed-samples-slider').addEventListener('input', (e) => {
            this.params.n_samples = parseInt(e.target.value);
            document.getElementById('embed-samples-val').textContent = this.params.n_samples;
            this.updateCode();
        });

        document.getElementById('embed-range-slider').addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.params.c_range = [-val, val];
            document.getElementById('embed-range-val').textContent = `[-${val}, ${val}]`;
            this.updateCode();
        });

        // Compute button
        document.getElementById('compute-embedding-btn').addEventListener('click', () => this.compute());

        // Auto-compute on ready
        MindFractal.onReady(() => this.compute());
    }

    updateCode() {
        let code;
        if (this.params.mode === 'param_space') {
            code = `from embeddings_core import compute_and_render_param_space
img = compute_and_render_param_space(n_samples=${this.params.n_samples})`;
        } else {
            code = `from embeddings_core import compute_and_render_embedding
img = compute_and_render_embedding(n_samples=${this.params.n_samples}, method='${this.params.mode}')`;
        }
        this.codeEl.textContent = code;
    }

    async compute() {
        this.statusEl.textContent = `Computing ${this.params.mode} embedding...`;
        this.imageEl.style.opacity = 0.5;

        try {
            let code, info;

            if (this.params.mode === 'param_space') {
                code = `
from embeddings_core import sample_trajectory_manifold, render_param_space_to_base64

features, params, labels = sample_trajectory_manifold(
    n_samples=${this.params.n_samples},
    c_range=(${this.params.c_range[0]}, ${this.params.c_range[1]})
)
render_param_space_to_base64(params, labels)
`;
                info = `<div class="info-row"><strong>View:</strong> Parameter Space Classification</div>
                        <div class="info-row"><strong>Samples:</strong> ${this.params.n_samples}</div>
                        <div class="info-row"><strong>c range:</strong> [${this.params.c_range[0]}, ${this.params.c_range[1]}]</div>`;
            } else {
                code = `
from embeddings_core import sample_trajectory_manifold, render_embedding_to_base64

features, params, labels = sample_trajectory_manifold(
    n_samples=${this.params.n_samples},
    c_range=(${this.params.c_range[0]}, ${this.params.c_range[1]})
)
render_embedding_to_base64(features, labels, method='${this.params.mode}')
`;
                const methodName = this.params.mode === 'pca' ? 'PCA' : 't-SNE';
                info = `<div class="info-row"><strong>Method:</strong> ${methodName}</div>
                        <div class="info-row"><strong>Samples:</strong> ${this.params.n_samples}</div>
                        <div class="info-row"><strong>Features:</strong> 12 per trajectory</div>`;
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
const embeddingsViewerStyles = `
<style>
.embeddings-viewer {
    background: linear-gradient(135deg, rgba(26, 26, 46, 0.95), rgba(22, 33, 62, 0.95));
    border-radius: 16px;
    padding: 24px;
    margin: 20px 0;
}

.embeddings-viewer .viewer-header h3 {
    color: #ffffff;
    margin: 0 0 8px 0;
}

.embeddings-viewer .viewer-header p {
    color: #a0a0a0;
    margin: 0 0 20px 0;
}

.embeddings-viewer .viewer-layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 24px;
}

@media (max-width: 768px) {
    .embeddings-viewer .viewer-layout {
        grid-template-columns: 1fr;
    }
}

.embeddings-viewer .control-section {
    margin-bottom: 20px;
}

.embeddings-viewer .control-section h4 {
    color: #06b6d4;
    margin: 0 0 12px 0;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.embeddings-viewer .mode-buttons {
    display: flex;
    gap: 8px;
}

.embeddings-viewer .mode-btn {
    flex: 1;
    padding: 8px 10px;
    background: rgba(124, 58, 237, 0.2);
    border: 1px solid #7c3aed;
    color: #a78bfa;
    border-radius: 6px;
    cursor: pointer;
    font-size: 11px;
    transition: all 0.2s;
}

.embeddings-viewer .mode-btn.active {
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    color: white;
    border-color: transparent;
}

.embeddings-viewer .control-group {
    margin-bottom: 12px;
}

.embeddings-viewer .control-group label {
    display: block;
    color: #e0e0e0;
    font-size: 13px;
    margin-bottom: 4px;
}

.embeddings-viewer input[type="range"] {
    width: 100%;
    height: 6px;
    background: #2d2250;
    border-radius: 3px;
    outline: none;
    -webkit-appearance: none;
}

.embeddings-viewer input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    border-radius: 50%;
    cursor: pointer;
}

.embeddings-viewer .primary-btn {
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

.embeddings-viewer .primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
}

.embeddings-viewer .info-box,
.embeddings-viewer .legend-box {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
}

.embeddings-viewer .info-box h4,
.embeddings-viewer .legend-box h4 {
    color: #06b6d4;
    margin: 0 0 8px 0;
    font-size: 12px;
}

.embeddings-viewer .info-box p {
    color: #a0a0a0;
    font-size: 12px;
    margin: 0;
    line-height: 1.5;
}

.embeddings-viewer .info-box ul {
    color: #a0a0a0;
    font-size: 11px;
    margin: 0;
    padding-left: 16px;
}

.embeddings-viewer .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    color: #e0e0e0;
    font-size: 12px;
}

.embeddings-viewer .legend-color {
    width: 12px;
    height: 12px;
    border-radius: 3px;
}

.embeddings-viewer .display-panel {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px;
    padding: 16px;
}

.embeddings-viewer .status-bar {
    color: #06b6d4;
    font-size: 13px;
    margin-bottom: 12px;
    padding: 8px;
    background: rgba(6, 182, 212, 0.1);
    border-radius: 6px;
}

.embeddings-viewer .image-container {
    text-align: center;
    margin-bottom: 16px;
}

.embeddings-viewer .result-image {
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    transition: opacity 0.3s;
}

.embeddings-viewer .info-panel {
    color: #c0c0c0;
    font-size: 13px;
}

.embeddings-viewer .info-row {
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.embeddings-viewer .code-panel {
    margin-top: 20px;
    padding: 16px;
    background: rgba(0, 0, 0, 0.4);
    border-radius: 8px;
}

.embeddings-viewer .code-panel h4 {
    color: #06b6d4;
    margin: 0 0 12px 0;
}

.embeddings-viewer .code-panel pre {
    margin: 0;
    padding: 12px;
    background: #0d1117;
    border-radius: 6px;
    overflow-x: auto;
}

.embeddings-viewer .code-panel code {
    color: #c9d1d9;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
</style>
`;

document.head.insertAdjacentHTML('beforeend', embeddingsViewerStyles);

// Export
window.EmbeddingsViewer = EmbeddingsViewer;

/**
 * Possibility Viewer — Possibility Manifold Navigator
 * MindFractal Lab
 *
 * Interactive exploration of the Possibility Manifold and timeline interpolation.
 */

class PossibilityViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container #${containerId} not found`);
            return;
        }

        // Parameters
        this.params = {
            mode: 'scan',  // 'scan', 'timeline', 'sample'
            resolution: 40,
            c1_min: -2,
            c1_max: 2,
            c2_min: -2,
            c2_max: 2,
            // Timeline points
            p1: { z0: [0.1, 0.1], c: [-0.5, 0.2] },
            p2: { z0: [0.1, 0.1], c: [0.5, -0.3] },
            n_timeline_points: 5
        };

        this.init();
    }

    init() {
        this.createUI();
        this.bindEvents();
    }

    createUI() {
        this.container.innerHTML = `
            <div class="possibility-viewer interactive-demo">
                <div class="viewer-header">
                    <h3>Possibility Manifold Navigator</h3>
                    <p class="explorer-only">Explore the space of possible dynamical states</p>
                </div>

                <div class="viewer-layout">
                    <div class="controls-panel">
                        <div class="control-section">
                            <h4>Exploration Mode</h4>
                            <div class="mode-buttons">
                                <button id="mode-scan" class="mode-btn active">Stability Scan</button>
                                <button id="mode-timeline" class="mode-btn">Timeline</button>
                                <button id="mode-sample" class="mode-btn">Sample</button>
                            </div>
                        </div>

                        <div class="control-section scan-controls">
                            <h4>Parameter Range</h4>
                            <div class="control-group">
                                <label>Resolution: <span id="scan-res-val">${this.params.resolution}</span></label>
                                <input type="range" id="scan-res-slider" min="20" max="80" step="10" value="${this.params.resolution}">
                            </div>
                            <div class="range-inputs">
                                <div class="range-row">
                                    <span>c₁:</span>
                                    <input type="number" id="c1-min" value="${this.params.c1_min}" step="0.5">
                                    <span>to</span>
                                    <input type="number" id="c1-max" value="${this.params.c1_max}" step="0.5">
                                </div>
                                <div class="range-row">
                                    <span>c₂:</span>
                                    <input type="number" id="c2-min" value="${this.params.c2_min}" step="0.5">
                                    <span>to</span>
                                    <input type="number" id="c2-max" value="${this.params.c2_max}" step="0.5">
                                </div>
                            </div>
                        </div>

                        <div class="control-section timeline-controls" style="display:none;">
                            <h4>Timeline Endpoints</h4>
                            <div class="endpoint-group">
                                <span class="endpoint-label">Start Point (t=0):</span>
                                <div class="control-row">
                                    <label>c₁:</label>
                                    <input type="number" id="p1-c1" value="${this.params.p1.c[0]}" step="0.1">
                                    <label>c₂:</label>
                                    <input type="number" id="p1-c2" value="${this.params.p1.c[1]}" step="0.1">
                                </div>
                            </div>
                            <div class="endpoint-group">
                                <span class="endpoint-label">End Point (t=1):</span>
                                <div class="control-row">
                                    <label>c₁:</label>
                                    <input type="number" id="p2-c1" value="${this.params.p2.c[0]}" step="0.1">
                                    <label>c₂:</label>
                                    <input type="number" id="p2-c2" value="${this.params.p2.c[1]}" step="0.1">
                                </div>
                            </div>
                            <div class="control-group">
                                <label>Timeline Points: <span id="tl-points-val">${this.params.n_timeline_points}</span></label>
                                <input type="range" id="tl-points-slider" min="3" max="10" step="1" value="${this.params.n_timeline_points}">
                            </div>
                        </div>

                        <div class="control-section sample-controls" style="display:none;">
                            <h4>Sampling</h4>
                            <div class="control-group">
                                <label>Number of Samples: <span id="n-samples-val">100</span></label>
                                <input type="range" id="n-samples-slider" min="50" max="500" step="50" value="100">
                            </div>
                            <div class="checkbox-group">
                                <label>
                                    <input type="checkbox" id="bounded-only" checked>
                                    Bounded trajectories only
                                </label>
                            </div>
                        </div>

                        <button id="compute-possibility-btn" class="primary-btn">Compute</button>
                    </div>

                    <div class="display-panel">
                        <div id="possibility-status" class="status-bar">Initializing...</div>
                        <div class="image-container">
                            <img id="possibility-image" class="result-image" alt="Possibility Manifold">
                        </div>
                        <div id="possibility-info" class="info-panel"></div>
                    </div>
                </div>

                <div class="researcher-only code-panel">
                    <h4>Python Code</h4>
                    <pre><code id="possibility-code">from possibility_core import scan_stability_region
result = scan_stability_region(resolution=${this.params.resolution})</code></pre>
                </div>
            </div>
        `;

        this.imageEl = document.getElementById('possibility-image');
        this.infoEl = document.getElementById('possibility-info');
        this.statusEl = document.getElementById('possibility-status');
        this.codeEl = document.getElementById('possibility-code');
    }

    bindEvents() {
        // Mode buttons
        ['scan', 'timeline', 'sample'].forEach(mode => {
            document.getElementById(`mode-${mode}`).addEventListener('click', (e) => {
                document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');
                this.params.mode = mode;
                this.updateControlVisibility();
                this.updateCode();
            });
        });

        // Scan controls
        document.getElementById('scan-res-slider').addEventListener('input', (e) => {
            this.params.resolution = parseInt(e.target.value);
            document.getElementById('scan-res-val').textContent = this.params.resolution;
            this.updateCode();
        });

        ['c1-min', 'c1-max', 'c2-min', 'c2-max'].forEach(id => {
            document.getElementById(id).addEventListener('change', (e) => {
                const key = id.replace('-', '_');
                this.params[key] = parseFloat(e.target.value);
                this.updateCode();
            });
        });

        // Timeline controls
        document.getElementById('p1-c1').addEventListener('change', (e) => {
            this.params.p1.c[0] = parseFloat(e.target.value);
            this.updateCode();
        });
        document.getElementById('p1-c2').addEventListener('change', (e) => {
            this.params.p1.c[1] = parseFloat(e.target.value);
            this.updateCode();
        });
        document.getElementById('p2-c1').addEventListener('change', (e) => {
            this.params.p2.c[0] = parseFloat(e.target.value);
            this.updateCode();
        });
        document.getElementById('p2-c2').addEventListener('change', (e) => {
            this.params.p2.c[1] = parseFloat(e.target.value);
            this.updateCode();
        });
        document.getElementById('tl-points-slider').addEventListener('input', (e) => {
            this.params.n_timeline_points = parseInt(e.target.value);
            document.getElementById('tl-points-val').textContent = this.params.n_timeline_points;
            this.updateCode();
        });

        // Compute button
        document.getElementById('compute-possibility-btn').addEventListener('click', () => this.compute());

        // Auto-compute on ready
        MindFractal.onReady(() => this.compute());
    }

    updateControlVisibility() {
        document.querySelector('.scan-controls').style.display =
            this.params.mode === 'scan' ? 'block' : 'none';
        document.querySelector('.timeline-controls').style.display =
            this.params.mode === 'timeline' ? 'block' : 'none';
        document.querySelector('.sample-controls').style.display =
            this.params.mode === 'sample' ? 'block' : 'none';
    }

    updateCode() {
        let code;
        if (this.params.mode === 'scan') {
            code = `from possibility_core import scan_stability_region
import matplotlib.pyplot as plt

result = scan_stability_region(
    resolution=${this.params.resolution},
    c1_range=(${this.params.c1_min}, ${this.params.c1_max}),
    c2_range=(${this.params.c2_min}, ${this.params.c2_max})
)
# result['lyapunov'] - Lyapunov exponent map
# result['classification'] - 0=stable, 1=periodic, 2=chaotic, 3=divergent`;
        } else if (this.params.mode === 'timeline') {
            code = `from possibility_core import PossibilityPoint, render_timeline_to_base64
import numpy as np

p1 = PossibilityPoint(
    z0=np.array([0.1, 0.1]),
    c=np.array([${this.params.p1.c[0]}, ${this.params.p1.c[1]}])
)
p2 = PossibilityPoint(
    z0=np.array([0.1, 0.1]),
    c=np.array([${this.params.p2.c[0]}, ${this.params.p2.c[1]}])
)
img = render_timeline_to_base64(p1, p2, n_points=${this.params.n_timeline_points})`;
        } else {
            code = `from possibility_core import sample_possibility_manifold

samples = sample_possibility_manifold(
    n_samples=100,
    bounded_only=True
)
# Returns list of PossibilityPoint objects`;
        }
        this.codeEl.textContent = code;
    }

    async compute() {
        this.statusEl.textContent = `Computing ${this.params.mode}...`;
        this.imageEl.style.opacity = 0.5;

        try {
            let code, info;

            if (this.params.mode === 'scan') {
                code = `
from possibility_core import scan_stability_region
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import base64
from io import BytesIO

result = scan_stability_region(
    resolution=${this.params.resolution},
    c1_range=(${this.params.c1_min}, ${this.params.c1_max}),
    c2_range=(${this.params.c2_min}, ${this.params.c2_max})
)

# Plot classification
colors = ['#3498db', '#f1c40f', '#e74c3c', '#1a1a2e']
cmap = ListedColormap(colors)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Lyapunov map
im1 = axes[0].imshow(result['lyapunov'], extent=[${this.params.c1_min}, ${this.params.c1_max}, ${this.params.c2_min}, ${this.params.c2_max}],
                      origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_xlabel('c₁', color='#cccccc')
axes[0].set_ylabel('c₂', color='#cccccc')
axes[0].set_title('Lyapunov Exponent', color='white')
plt.colorbar(im1, ax=axes[0])

# Classification map
im2 = axes[1].imshow(result['classification'], extent=[${this.params.c1_min}, ${this.params.c1_max}, ${this.params.c2_min}, ${this.params.c2_max}],
                      origin='lower', cmap=cmap, vmin=0, vmax=3)
axes[1].set_xlabel('c₁', color='#cccccc')
axes[1].set_ylabel('c₂', color='#cccccc')
axes[1].set_title('Classification', color='white')

for ax in axes:
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#cccccc')
fig.patch.set_facecolor('#1a1a2e')

plt.tight_layout()
buf = BytesIO()
fig.savefig(buf, format='png', dpi=100, facecolor='#1a1a2e')
plt.close(fig)
buf.seek(0)
base64.b64encode(buf.read()).decode('utf-8')
`;
                info = `<div class="info-row"><strong>Mode:</strong> Stability Scan</div>
                        <div class="info-row"><strong>Range:</strong> c₁ ∈ [${this.params.c1_min}, ${this.params.c1_max}], c₂ ∈ [${this.params.c2_min}, ${this.params.c2_max}]</div>
                        <div class="info-row">
                            <span style="color:#3498db;">Stable</span> |
                            <span style="color:#f1c40f;">Periodic</span> |
                            <span style="color:#e74c3c;">Chaotic</span>
                        </div>`;

            } else if (this.params.mode === 'timeline') {
                code = `
from possibility_core import PossibilityPoint, render_timeline_to_base64
import numpy as np

p1 = PossibilityPoint(
    z0=np.array([0.1, 0.1]),
    c=np.array([${this.params.p1.c[0]}, ${this.params.p1.c[1]}])
)
p2 = PossibilityPoint(
    z0=np.array([0.1, 0.1]),
    c=np.array([${this.params.p2.c[0]}, ${this.params.p2.c[1]}])
)
render_timeline_to_base64(p1, p2, n_points=${this.params.n_timeline_points})
`;
                info = `<div class="info-row"><strong>Mode:</strong> Timeline Interpolation</div>
                        <div class="info-row"><strong>Start:</strong> c = (${this.params.p1.c[0]}, ${this.params.p1.c[1]})</div>
                        <div class="info-row"><strong>End:</strong> c = (${this.params.p2.c[0]}, ${this.params.p2.c[1]})</div>
                        <div class="info-row"><strong>Points:</strong> ${this.params.n_timeline_points}</div>`;

            } else {
                const nSamples = parseInt(document.getElementById('n-samples-slider').value);
                const boundedOnly = document.getElementById('bounded-only').checked;
                code = `
from possibility_core import sample_possibility_manifold, classify_point
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

samples = sample_possibility_manifold(n_samples=${nSamples}, bounded_only=${boundedOnly ? 'True' : 'False'})

# Plot samples in parameter space
fig, ax = plt.subplots(figsize=(8, 8))

label_colors = {'stable': '#3498db', 'periodic': '#f1c40f', 'chaotic': '#e74c3c', 'divergent': '#7f8c8d'}

for p in samples:
    label = classify_point(p)
    color = label_colors.get(label, '#888888')
    ax.scatter(p.c[0], p.c[1], c=color, s=30, alpha=0.7)

ax.set_xlabel('c₁', color='#cccccc', fontsize=12)
ax.set_ylabel('c₂', color='#cccccc', fontsize=12)
ax.set_title('Sampled Possibility Points', color='white', fontsize=14)
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#1a1a2e')
ax.tick_params(colors='#cccccc')

# Legend
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=l)
           for l, c in label_colors.items()]
ax.legend(handles=handles, loc='upper right')

plt.tight_layout()
buf = BytesIO()
fig.savefig(buf, format='png', dpi=100, facecolor='#1a1a2e')
plt.close(fig)
buf.seek(0)
base64.b64encode(buf.read()).decode('utf-8')
`;
                info = `<div class="info-row"><strong>Mode:</strong> Random Sampling</div>
                        <div class="info-row"><strong>Samples:</strong> ${nSamples}</div>
                        <div class="info-row"><strong>Bounded Only:</strong> ${boundedOnly ? 'Yes' : 'No'}</div>`;
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
const possibilityViewerStyles = `
<style>
.possibility-viewer {
    background: linear-gradient(135deg, rgba(26, 26, 46, 0.95), rgba(22, 33, 62, 0.95));
    border-radius: 16px;
    padding: 24px;
    margin: 20px 0;
}

.possibility-viewer .viewer-header h3 {
    color: #ffffff;
    margin: 0 0 8px 0;
}

.possibility-viewer .viewer-header p {
    color: #a0a0a0;
    margin: 0 0 20px 0;
}

.possibility-viewer .viewer-layout {
    display: grid;
    grid-template-columns: 300px 1fr;
    gap: 24px;
}

@media (max-width: 768px) {
    .possibility-viewer .viewer-layout {
        grid-template-columns: 1fr;
    }
}

.possibility-viewer .control-section {
    margin-bottom: 20px;
}

.possibility-viewer .control-section h4 {
    color: #06b6d4;
    margin: 0 0 12px 0;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.possibility-viewer .mode-buttons {
    display: flex;
    gap: 8px;
}

.possibility-viewer .mode-btn {
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

.possibility-viewer .mode-btn.active {
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    color: white;
    border-color: transparent;
}

.possibility-viewer .control-group {
    margin-bottom: 12px;
}

.possibility-viewer .control-group label {
    display: block;
    color: #e0e0e0;
    font-size: 13px;
    margin-bottom: 4px;
}

.possibility-viewer input[type="range"] {
    width: 100%;
    height: 6px;
    background: #2d2250;
    border-radius: 3px;
    outline: none;
    -webkit-appearance: none;
}

.possibility-viewer input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    border-radius: 50%;
    cursor: pointer;
}

.possibility-viewer .range-inputs {
    margin-top: 10px;
}

.possibility-viewer .range-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    color: #e0e0e0;
    font-size: 13px;
}

.possibility-viewer .range-row input[type="number"] {
    width: 60px;
    padding: 6px;
    background: #2d2250;
    border: 1px solid #7c3aed;
    color: #e0e0e0;
    border-radius: 4px;
    font-size: 12px;
}

.possibility-viewer .endpoint-group {
    margin-bottom: 12px;
}

.possibility-viewer .endpoint-label {
    display: block;
    color: #a0a0a0;
    font-size: 12px;
    margin-bottom: 6px;
}

.possibility-viewer .control-row {
    display: flex;
    align-items: center;
    gap: 8px;
}

.possibility-viewer .control-row label {
    color: #e0e0e0;
    font-size: 12px;
    margin: 0;
}

.possibility-viewer .control-row input[type="number"] {
    width: 60px;
    padding: 6px;
    background: #2d2250;
    border: 1px solid #7c3aed;
    color: #e0e0e0;
    border-radius: 4px;
    font-size: 12px;
}

.possibility-viewer .checkbox-group {
    margin-top: 10px;
}

.possibility-viewer .checkbox-group label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #e0e0e0;
    font-size: 13px;
    cursor: pointer;
}

.possibility-viewer .primary-btn {
    width: 100%;
    padding: 12px;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    transition: transform 0.2s, box-shadow 0.2s;
}

.possibility-viewer .primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
}

.possibility-viewer .display-panel {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px;
    padding: 16px;
}

.possibility-viewer .status-bar {
    color: #06b6d4;
    font-size: 13px;
    margin-bottom: 12px;
    padding: 8px;
    background: rgba(6, 182, 212, 0.1);
    border-radius: 6px;
}

.possibility-viewer .image-container {
    text-align: center;
    margin-bottom: 16px;
}

.possibility-viewer .result-image {
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    transition: opacity 0.3s;
}

.possibility-viewer .info-panel {
    color: #c0c0c0;
    font-size: 13px;
}

.possibility-viewer .info-row {
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.possibility-viewer .code-panel {
    margin-top: 20px;
    padding: 16px;
    background: rgba(0, 0, 0, 0.4);
    border-radius: 8px;
}

.possibility-viewer .code-panel h4 {
    color: #06b6d4;
    margin: 0 0 12px 0;
}

.possibility-viewer .code-panel pre {
    margin: 0;
    padding: 12px;
    background: #0d1117;
    border-radius: 6px;
    overflow-x: auto;
}

.possibility-viewer .code-panel code {
    color: #c9d1d9;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
</style>
`;

document.head.insertAdjacentHTML('beforeend', possibilityViewerStyles);

// Export
window.PossibilityViewer = PossibilityViewer;

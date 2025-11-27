# Spacetime Weave Explorer

<main class="mf-module-page">
<header class="mf-module-header">
<h2>Quantum Gravity — Emergent Geometry</h2>
<p class="mf-module-description">
Watch spacetime itself emerge from quantum fluctuations. Each node represents a quantum of geometry;
their connections weave the fabric of space. Adjust coherence to see order arise from chaos.
</p>
</header>

<section class="mf-module-layout">
<aside class="mf-controls-panel" id="qg-controls">
<h3>Parameters</h3>

<div class="mf-control-group">
<label for="node-count">Nodes</label>
<input type="range" id="node-count" min="25" max="400" step="25" value="100">
<span id="node-count-value">100</span>
</div>

<div class="mf-control-group">
<label for="noise-level">Quantum Noise</label>
<input type="range" id="noise-level" min="0" max="1" step="0.05" value="0.3">
<span id="noise-level-value">0.3</span>
</div>

<div class="mf-control-group">
<label for="coherence">Coherence</label>
<input type="range" id="coherence" min="0" max="1" step="0.05" value="0.5">
<span id="coherence-value">0.5</span>
</div>

<div class="mf-control-group">
<label for="curvature">Curvature Bias</label>
<input type="range" id="curvature" min="-1" max="1" step="0.1" value="0">
<span id="curvature-value">0</span>
</div>

<div class="mf-button-group">
<button id="animate-btn" class="mf-btn">Animate</button>
<button id="randomize-btn" class="mf-btn mf-btn-secondary">Randomize</button>
</div>
</aside>

<section class="mf-canvas-panel">
<canvas id="quantum-gravity-canvas" width="800" height="600"></canvas>
<div id="qg-status" class="mf-status"></div>
</section>
</section>

<section class="mf-module-explainer">
<h3>What You're Seeing</h3>
<p>
This visualization draws on loop quantum gravity and spin foam models, where spacetime is not a smooth
continuum but a network of discrete quanta. The nodes represent Planck-scale geometric elements;
their connections carry information about area and volume.
</p>

<h4>The Controls</h4>
<ul>
<li><strong>Nodes:</strong> Number of spacetime quanta in the weave</li>
<li><strong>Quantum Noise:</strong> Intensity of Planck-scale fluctuations — higher values show spacetime "foam"</li>
<li><strong>Coherence:</strong> How ordered the geometry is — at low coherence, the structure is chaotic</li>
<li><strong>Curvature Bias:</strong> Positive = nodes curve inward (like near a mass); Negative = nodes spread outward</li>
</ul>

<h4>Try This</h4>
<ol>
<li>Set coherence to 0 — observe pure quantum chaos</li>
<li>Slowly increase coherence — watch smooth geometry emerge</li>
<li>Add positive curvature — see how mass would warp this space</li>
<li>Press "Animate" to see the foam fluctuate in time</li>
</ol>
</section>
</main>

<script type="module">
import { QuantumGravityViewer } from '../site/interactive/js/quantum_gravity_viewer.js';
</script>

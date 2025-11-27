# Implicate/Explicate Projector

<main class="mf-module-page">
<header class="mf-module-header">
<h2>Holographic Universe — Boundary to Bulk</h2>
<p class="mf-module-description">
Information encoded on a lower-dimensional boundary projects into the higher-dimensional bulk.
Paint on the implicate order and watch the explicate reality emerge.
</p>
</header>

<section class="mf-module-layout">
<aside class="mf-controls-panel" id="hu-controls">
<h3>Projection Parameters</h3>

<div class="mf-control-group">
<label for="projection-depth">Projection Depth</label>
<input type="range" id="projection-depth" min="0.1" max="1" step="0.05" value="0.5">
<span id="projection-depth-value">0.5</span>
</div>

<div class="mf-control-group">
<label for="smoothness">Smoothness</label>
<input type="range" id="smoothness" min="0" max="1" step="0.05" value="0.3">
<span id="smoothness-value">0.3</span>
</div>

<div class="mf-control-group">
<label for="resolution">Resolution</label>
<input type="range" id="resolution" min="30" max="80" step="10" value="60">
<span id="resolution-value">60</span>
</div>

<h3>Boundary Encoding</h3>
<div class="mf-button-group mf-encoding-btns">
<button id="enc-wave" class="mf-btn mf-btn-small encoding-btn active">Wave</button>
<button id="enc-spiral" class="mf-btn mf-btn-small encoding-btn">Spiral</button>
<button id="enc-stripes" class="mf-btn mf-btn-small encoding-btn">Stripes</button>
<button id="enc-noise" class="mf-btn mf-btn-small encoding-btn">Noise</button>
<button id="enc-checkerboard" class="mf-btn mf-btn-small encoding-btn">Check</button>
</div>

<h3>View Mode</h3>
<div class="mf-button-group">
<button id="view-split" class="mf-btn view-btn active">Split</button>
<button id="view-boundary" class="mf-btn view-btn">Boundary</button>
<button id="view-explicate" class="mf-btn view-btn">Bulk</button>
<button id="view-overlay" class="mf-btn view-btn">Overlay</button>
</div>

<h3>Paint on Boundary</h3>
<div class="mf-control-group">
<button id="brush-toggle" class="mf-btn">Brush: OFF</button>
</div>
<div class="mf-control-group">
<label for="brush-value">Brush Value</label>
<input type="range" id="brush-value" min="0" max="1" step="0.1" value="1">
</div>

<div class="mf-button-group">
<button id="randomize-btn" class="mf-btn mf-btn-secondary">Randomize</button>
</div>
</aside>

<section class="mf-canvas-panel">
<canvas id="holographic-universe-canvas" width="800" height="600"></canvas>
<div id="hu-status" class="mf-status"></div>
</section>
</section>

<section class="mf-module-explainer">
<h3>The Holographic Principle</h3>
<p>
In theoretical physics, the holographic principle proposes that all information contained in a volume
of space can be encoded on its boundary. This visualization demonstrates the relationship between
boundary information (implicate order) and bulk patterns (explicate order).
</p>

<h4>Understanding the Views</h4>
<ul>
<li><strong>Split View:</strong> Left shows boundary encoding (implicate); right shows projected bulk (explicate)</li>
<li><strong>Overlay:</strong> Both layers superimposed — red/magenta = boundary, cyan = bulk</li>
<li><strong>Entropy bars:</strong> Compare information content between boundary and bulk</li>
</ul>

<h4>David Bohm's Orders</h4>
<p>
Physicist David Bohm proposed that reality has two aspects:
</p>
<ul>
<li><strong>Implicate Order:</strong> The hidden, enfolded information substrate</li>
<li><strong>Explicate Order:</strong> The unfolded, manifest reality we perceive</li>
</ul>
<p>
This visualization lets you see how changing the implicate (boundary) instantly transforms the explicate (bulk).
</p>

<h4>Experiment</h4>
<ol>
<li>Try different encoding patterns — see how each creates different bulk structures</li>
<li>Adjust projection depth — deeper projection creates more complex bulk patterns</li>
<li>Enable the brush and paint on the boundary — watch the bulk transform in real-time</li>
<li>Compare entropies — does the bulk have more or less information than the boundary?</li>
</ol>
</section>
</main>

<script type="module">
import { HolographicUniverseViewer } from '../site/interactive/js/holographic_universe_viewer.js';
</script>

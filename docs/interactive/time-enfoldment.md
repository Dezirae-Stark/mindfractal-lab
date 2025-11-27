# Time-Enfoldment Navigator

<main class="mf-module-page">
<header class="mf-module-header">
<h2>Bidirectional Timeline — Past and Future Enfolded</h2>
<p class="mf-module-description">
Explore a timeline where past and future are not separate — they influence each other.
The present moment sits at the center; branches of possibility extend in both directions.
</p>
</header>

<section class="mf-module-layout">
<aside class="mf-controls-panel" id="te-controls">
<h3>Timeline Parameters</h3>

<div class="mf-control-group">
<label for="depth">Depth</label>
<input type="range" id="depth" min="1" max="8" step="1" value="4">
<span id="depth-value">4</span>
</div>

<div class="mf-control-group">
<label for="branching">Branching Factor</label>
<input type="range" id="branching" min="1" max="4" step="0.5" value="2">
<span id="branching-value">2</span>
</div>

<div class="mf-control-group">
<label for="retro-weight">Retrocausal Influence</label>
<input type="range" id="retro-weight" min="0" max="1" step="0.1" value="0.3">
<span id="retro-weight-value">0.3</span>
</div>

<div class="mf-control-group">
<label for="decoherence">Decoherence</label>
<input type="range" id="decoherence" min="0" max="1" step="0.1" value="0.2">
<span id="decoherence-value">0.2</span>
</div>

<h3>Decision Impact</h3>
<div class="mf-control-group">
<label for="decision-strength">Decision Strength</label>
<input type="range" id="decision-strength" min="0" max="1" step="0.1" value="0">
<span id="decision-strength-value">0</span>
</div>

<div class="mf-button-group">
<button id="randomize-btn" class="mf-btn mf-btn-secondary">New Timeline</button>
</div>
</aside>

<section class="mf-canvas-panel">
<canvas id="time-enfoldment-canvas" width="800" height="600"></canvas>
<div id="te-status" class="mf-status"></div>
</section>
</section>

<section class="mf-module-explainer">
<h3>Understanding the Timeline</h3>
<p>
This model is inspired by Wheeler-Feynman absorber theory and retrocausal interpretations of quantum mechanics.
In these frameworks, the future is not merely a consequence of the past — it can influence the present
just as the past does.
</p>

<h4>Reading the Visualization</h4>
<ul>
<li><strong>Center (NOW):</strong> The present moment — the bright node at the center</li>
<li><strong>Left (Purple):</strong> Past branches — events that have already collapsed</li>
<li><strong>Right (Cyan):</strong> Future branches — possibilities yet to manifest</li>
<li><strong>Node brightness:</strong> Probability weight of that timeline branch</li>
</ul>

<h4>The Controls</h4>
<ul>
<li><strong>Retrocausal Influence:</strong> How much the future affects the present structure</li>
<li><strong>Decoherence:</strong> How quickly branches fade from influence</li>
<li><strong>Decision Strength:</strong> Simulate making a choice — watch the future branches multiply</li>
</ul>

<h4>Experiment</h4>
<ol>
<li>Increase retrocausal influence — see how future structure affects past branching</li>
<li>Slide the decision strength — observe the timeline reshaping in real-time</li>
<li>Hover over nodes to see their probability weights</li>
<li>Click the center node to reset</li>
</ol>
</section>
</main>

<script type="module">
import { TimeEnfoldmentViewer } from '../site/interactive/js/time_enfoldment_viewer.js';
</script>

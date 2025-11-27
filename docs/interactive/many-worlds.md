# Branching Universe Explorer

<main class="mf-module-page">
<header class="mf-module-header">
<h2>Many-Worlds — Quantum Branching</h2>
<p class="mf-module-description">
Every quantum measurement splits reality into branches. Explore the tree of possibilities,
select a path through the multiverse, and see interference patterns emerge from coherent branches.
</p>
</header>

<section class="mf-module-layout">
<aside class="mf-controls-panel" id="mw-controls">
<h3>Branching Parameters</h3>

<div class="mf-control-group">
<label for="depth">Tree Depth</label>
<input type="range" id="depth" min="2" max="8" step="1" value="5">
<span id="depth-value">5</span>
</div>

<div class="mf-control-group">
<label for="branching">Branching Factor</label>
<input type="range" id="branching" min="1.5" max="4" step="0.25" value="2">
<span id="branching-value">2</span>
</div>

<div class="mf-control-group">
<label for="decoherence">Decoherence Rate</label>
<input type="range" id="decoherence" min="0" max="0.5" step="0.05" value="0.1">
<span id="decoherence-value">0.1</span>
</div>

<div class="mf-control-group">
<label for="prob-compression">Probability Compression</label>
<input type="range" id="prob-compression" min="0" max="0.8" step="0.1" value="0.3">
<span id="prob-compression-value">0.3</span>
</div>

<h3>View Mode</h3>
<div class="mf-button-group">
<button id="view-tree" class="mf-btn active">Branch Tree</button>
<button id="view-interference" class="mf-btn">Interference</button>
</div>

<h3>Interference Settings</h3>
<div class="mf-control-group">
<label for="n-branches">Interfering Branches</label>
<input type="range" id="n-branches" min="2" max="8" step="1" value="4">
<span id="n-branches-value">4</span>
</div>

<div class="mf-control-group">
<label for="int-coherence">Coherence</label>
<input type="range" id="int-coherence" min="0" max="1" step="0.1" value="0.5">
<span id="int-coherence-value">0.5</span>
</div>

<div class="mf-button-group">
<button id="randomize-btn" class="mf-btn mf-btn-secondary">New Universe</button>
<button id="clear-selection" class="mf-btn mf-btn-secondary">Clear Path</button>
</div>
</aside>

<section class="mf-canvas-panel">
<canvas id="many-worlds-canvas" width="800" height="600"></canvas>
<div id="mw-status" class="mf-status"></div>
</section>
</section>

<section class="mf-module-explainer">
<h3>The Many-Worlds Interpretation</h3>
<p>
In Hugh Everett's many-worlds interpretation of quantum mechanics, the wave function never collapses.
Instead, every quantum measurement causes the universe to branch. All outcomes occur — each in its own
branch of reality.
</p>

<h4>Reading the Branch Tree</h4>
<ul>
<li><strong>Root (left):</strong> The initial state before branching</li>
<li><strong>Branches:</strong> Each split represents a quantum measurement outcome</li>
<li><strong>Node brightness:</strong> Probability amplitude of that branch</li>
<li><strong>Selected path (gold):</strong> Click nodes to trace a specific world-line</li>
</ul>

<h4>The Controls</h4>
<ul>
<li><strong>Branching Factor:</strong> Average number of outcomes per measurement</li>
<li><strong>Decoherence:</strong> How quickly branches lose coherence with each other</li>
<li><strong>Probability Compression:</strong> Whether probability concentrates in few branches or spreads evenly</li>
</ul>

<h4>Interference View</h4>
<p>
When branches remain coherent, they can interfere — their probability amplitudes add constructively
or destructively. The interference pattern shows the combined effect of multiple branches.
</p>
<ul>
<li><strong>High coherence:</strong> Strong interference fringes (quantum behavior)</li>
<li><strong>Low coherence:</strong> Washed-out pattern (classical behavior)</li>
</ul>

<h4>Experiment</h4>
<ol>
<li>Click on a leaf node — trace your path through the multiverse</li>
<li>Increase decoherence — watch branches fade as they lose quantum coherence</li>
<li>Switch to Interference view — see how multiple branches combine</li>
<li>Adjust coherence in interference mode — observe quantum-to-classical transition</li>
</ol>
</section>
</main>

<script type="module">
import { ManyWorldsViewer } from '../site/interactive/js/many_worlds_viewer.js';
</script>

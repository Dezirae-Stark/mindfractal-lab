# Coherence & Resonance Map

<main class="mf-module-page">
<header class="mf-module-header">
<h2>Gateway Coherence — Hemispheric Synchronization</h2>
<p class="mf-module-description">
Two oscillators, representing left and right hemispheres of awareness, dance toward synchrony.
Watch phase-locking emerge as coupling increases — the signature of coherent consciousness.
</p>
</header>

<section class="mf-module-layout">
<aside class="mf-controls-panel" id="gc-controls">
<h3>Oscillator Parameters</h3>

<div class="mf-control-group">
<label for="freq-left">Left Frequency</label>
<input type="range" id="freq-left" min="0.5" max="5" step="0.1" value="1.0">
<span id="freq-left-value">1.00</span>
</div>

<div class="mf-control-group">
<label for="freq-right">Right Frequency</label>
<input type="range" id="freq-right" min="0.5" max="5" step="0.1" value="1.0">
<span id="freq-right-value">1.00</span>
</div>

<div class="mf-control-group">
<label for="coupling">Coupling Strength</label>
<input type="range" id="coupling" min="0" max="1" step="0.05" value="0.5">
<span id="coupling-value">0.50</span>
</div>

<div class="mf-control-group">
<label for="phase-offset">Phase Offset</label>
<input type="range" id="phase-offset" min="0" max="6.28" step="0.1" value="0">
<span id="phase-offset-value">0.00</span>
</div>

<div class="mf-control-group">
<label for="noise">Noise Level</label>
<input type="range" id="noise" min="0" max="0.5" step="0.05" value="0.1">
<span id="noise-value">0.10</span>
</div>

<h3>View Mode</h3>
<div class="mf-button-group">
<button id="view-pattern" class="mf-btn active">Pattern</button>
<button id="view-field" class="mf-btn">Field</button>
</div>

<h3>Presets</h3>
<div class="mf-button-group mf-presets">
<button class="mf-btn mf-btn-small binaural-preset" data-preset="delta">Delta</button>
<button class="mf-btn mf-btn-small binaural-preset" data-preset="theta">Theta</button>
<button class="mf-btn mf-btn-small binaural-preset" data-preset="alpha">Alpha</button>
<button class="mf-btn mf-btn-small binaural-preset" data-preset="beta">Beta</button>
<button class="mf-btn mf-btn-small binaural-preset" data-preset="gamma">Gamma</button>
</div>

<div class="mf-button-group">
<button id="animate-btn" class="mf-btn mf-btn-secondary">Animate</button>
</div>
</aside>

<section class="mf-canvas-panel">
<canvas id="gateway-coherence-canvas" width="800" height="600"></canvas>
<div id="gc-status" class="mf-status"></div>
</section>
</section>

<section class="mf-module-explainer">
<h3>The Science of Coherence</h3>
<p>
This visualization is based on coupled oscillator dynamics (Kuramoto model) and concepts from
hemispheric synchronization research. When two oscillators with similar frequencies are coupled,
they tend toward phase-locking — a state where they oscillate in sync.
</p>

<h4>Reading the Pattern</h4>
<ul>
<li><strong>Pattern View:</strong> A Lissajous-like trace showing the combined oscillation</li>
<li><strong>Field View:</strong> A 2D interference pattern showing resonance peaks</li>
<li><strong>Coherence Meter:</strong> Measures how synchronized the oscillators are (0-100%)</li>
<li><strong>Phase Lock:</strong> Achieved when coherence is high and frequencies are close</li>
</ul>

<h4>Brainwave Presets</h4>
<p>These presets correspond to different states of consciousness:</p>
<ul>
<li><strong>Delta (1 Hz):</strong> Deep sleep, unconscious processing</li>
<li><strong>Theta (4-8 Hz):</strong> Meditation, creativity, REM sleep</li>
<li><strong>Alpha (8-12 Hz):</strong> Relaxed alertness, calm focus</li>
<li><strong>Beta (12-30 Hz):</strong> Active thinking, problem solving</li>
<li><strong>Gamma (30+ Hz):</strong> Peak awareness, insight, binding</li>
</ul>

<h4>Experiment</h4>
<ol>
<li>Set both frequencies equal — observe tight, coherent patterns</li>
<li>Slightly detune the right frequency — watch the pattern destabilize</li>
<li>Increase coupling — see how the oscillators pull back into sync</li>
<li>Switch to Field view to see the resonance structure in 2D</li>
</ol>
</section>
</main>

<script type="module">
import { GatewayCoherenceViewer } from '../site/interactive/js/gateway_coherence_viewer.js';
</script>

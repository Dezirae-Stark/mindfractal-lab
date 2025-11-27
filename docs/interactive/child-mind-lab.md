# Child Mind Lab

<div class="explorer-container child-mind-lab">
    <div class="explorer-header">
        <h2 class="explorer-title">Child Mind v1</h2>
        <p class="explorer-subtitle">Synthetic Agent on a Consciousness Manifold</p>
    </div>

    <div class="explorer-description">
        <p>
            Watch a simplified "mind" navigate its own internal space. The agent exists
            on a mathematical manifold where its position represents cognitive state—
            abstract thoughts, memories, and awareness all encoded as geometry.
        </p>
        <p>
            <strong>The bright dot is the agent's current "thought."</strong> As it moves,
            trails show where it has been. Colors shift from <span style="color: #ff4fa3;">pink</span>
            (low coherence) through <span style="color: #c45aff;">purple</span> to
            <span style="color: #39d8ff;">cyan</span> (high coherence) as the agent's
            internal consistency changes.
        </p>
    </div>

    <div class="explorer-main">
        <div class="canvas-container">
            <canvas id="child-mind-canvas" width="500" height="500"></canvas>
            <div class="canvas-status" id="child-mind-status">Initializing...</div>
        </div>

        <div class="controls-panel">
            <h3 class="controls-title">Agent Controls</h3>

            <div class="control-group">
                <label class="control-label">
                    <span>Curiosity</span>
                    <span class="control-value" id="curiosity-value">0.50</span>
                </label>
                <input type="range" id="curiosity-slider" class="control-slider"
                       min="0" max="1" step="0.01" value="0.5">
                <p class="control-hint">Higher curiosity = larger explorations, occasional jumps</p>
            </div>

            <div class="control-group">
                <label class="control-label">
                    <span>Coherence Preference</span>
                    <span class="control-value" id="coherence-pref-value">0.50</span>
                </label>
                <input type="range" id="coherence-pref-slider" class="control-slider"
                       min="0" max="1" step="0.01" value="0.5">
                <p class="control-hint">Higher preference = agent seeks internal consistency</p>
            </div>

            <div class="button-group">
                <button id="step-btn" class="control-button primary">
                    <span class="btn-icon">⏭</span> Step
                </button>
                <button id="auto-run-btn" class="control-button">
                    <span class="btn-icon">▶</span> Auto
                </button>
                <button id="reset-btn" class="control-button">
                    <span class="btn-icon">↺</span> Reset
                </button>
            </div>
        </div>
    </div>

    <div class="metrics-panel">
        <h3 class="metrics-title">Agent State</h3>
        <div class="metrics-grid">
            <div class="metric-card">
                <span class="metric-label">Coherence</span>
                <span class="metric-value" id="coherence-value">—</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Stability</span>
                <span class="metric-value" id="stability-value">—</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Novelty</span>
                <span class="metric-value" id="novelty-value">—</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Reward</span>
                <span class="metric-value" id="reward-value">—</span>
            </div>
            <div class="metric-card wide">
                <span class="metric-label">Time Step</span>
                <span class="metric-value" id="step-count">0</span>
            </div>
        </div>
    </div>
</div>

<div class="theory-section">
    <h2>How It Works</h2>

    <div class="theory-card">
        <h3>The Manifold Mind</h3>
        <p>
            The agent's "mind" lives on a 16-dimensional manifold. Each point in this space
            represents a possible cognitive configuration—think of it as the space of all
            possible thoughts the agent could have. The visualization projects two of these
            dimensions so you can watch the agent's journey.
        </p>
    </div>

    <div class="theory-card">
        <h3>Coherence & The Reward</h3>
        <p>
            The agent seeks coherence—internal consistency of its thoughts. Too little
            coherence and thoughts become fragmented; too much and they become rigid.
            The reward function balances:
        </p>
        <ul>
            <li><strong>Coherence:</strong> Staying near a target "sweet spot"</li>
            <li><strong>Novelty:</strong> Exploring new regions of thought-space</li>
            <li><strong>Stability:</strong> Avoiding erratic jumps</li>
        </ul>
    </div>

    <div class="theory-card">
        <h3>The Boundary</h3>
        <p>
            Inspired by holographic theories of consciousness, the agent's state connects
            to a 16×16 boundary grid. Actions don't just move the agent—they subtly rewrite
            patterns on this boundary, which in turn influence future dynamics. This creates
            a feedback loop between "bulk" cognition and "boundary" constraints.
        </p>
    </div>

    <div class="theory-card">
        <h3>Memory & Learning</h3>
        <p>
            The agent maintains a memory summary that tracks recent experiences. This isn't
            full episodic memory—it's more like an exponential moving average of where the
            agent has been and what actions it has taken. The memory influences how the
            agent evaluates novelty: visiting truly new regions of thought-space is rewarded.
        </p>
    </div>
</div>

<div class="interpretation-section">
    <h2>Interpreting the Visualization</h2>

    <div class="interpretation-grid">
        <div class="interpretation-item">
            <div class="interpretation-icon" style="background: linear-gradient(135deg, #ff4fa3, #c45aff);">●</div>
            <div class="interpretation-text">
                <strong>Pink/Purple Trails:</strong> Lower coherence. The agent's internal
                state is less organized, possibly exploring or transitioning.
            </div>
        </div>

        <div class="interpretation-item">
            <div class="interpretation-icon" style="background: linear-gradient(135deg, #c45aff, #39d8ff);">●</div>
            <div class="interpretation-text">
                <strong>Purple/Cyan Trails:</strong> Higher coherence. The agent has found
                a more integrated state—thoughts are consistent and stable.
            </div>
        </div>

        <div class="interpretation-item">
            <div class="interpretation-icon" style="background: #ffffff;">●</div>
            <div class="interpretation-text">
                <strong>White Point:</strong> Current position. The size of the outer ring
                indicates stability—larger rings mean the agent is in a stable attractor.
            </div>
        </div>

        <div class="interpretation-item">
            <div class="interpretation-icon" style="background: transparent; border: 2px solid #6b6bff;">○</div>
            <div class="interpretation-text">
                <strong>Boundary Circle:</strong> The region where most trajectories remain.
                Moving near the edge indicates extreme cognitive states.
            </div>
        </div>
    </div>
</div>

<div class="experiment-section">
    <h2>Things to Try</h2>

    <div class="experiment-grid">
        <div class="experiment-card">
            <h4>High Curiosity Mode</h4>
            <p>
                Set curiosity to 0.9+. Watch the agent make larger jumps and occasionally
                leap to entirely new regions. Notice how coherence fluctuates more wildly.
            </p>
        </div>

        <div class="experiment-card">
            <h4>Coherence Seeking</h4>
            <p>
                Set coherence preference to 0.8+. The agent will tend toward stable,
                consistent states—look for tighter, more circular trajectories.
            </p>
        </div>

        <div class="experiment-card">
            <h4>Balance Point</h4>
            <p>
                Try curiosity at 0.5, coherence at 0.7. This often produces the highest
                average reward—the agent explores enough to find good states but doesn't
                destabilize itself.
            </p>
        </div>

        <div class="experiment-card">
            <h4>Reset & Compare</h4>
            <p>
                Reset multiple times with the same settings. Each run uses a different
                random seed—notice how trajectories differ despite identical parameters.
                This is the signature of a complex dynamical system.
            </p>
        </div>
    </div>
</div>

<style>
.child-mind-lab {
    max-width: 1000px;
    margin: 0 auto;
}

.child-mind-lab .canvas-container {
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
    background: var(--bg-surface);
    border-radius: 12px;
    padding: 1rem;
}

.child-mind-lab .canvas-status {
    position: absolute;
    bottom: 1.5rem;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(10, 10, 26, 0.8);
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.75rem;
    color: var(--text-secondary);
}

.child-mind-lab #child-mind-canvas {
    border-radius: 8px;
    box-shadow: 0 0 30px rgba(107, 107, 255, 0.2);
}

.metrics-panel {
    margin-top: 2rem;
    padding: 1.5rem;
    background: var(--bg-surface);
    border-radius: 12px;
    border: 1px solid var(--border-color);
}

.metrics-title {
    margin: 0 0 1rem;
    font-size: 1rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1rem;
}

.metric-card {
    background: var(--bg-card);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    border: 1px solid var(--border-color);
}

.metric-card.wide {
    grid-column: span 2;
}

.metric-label {
    display: block;
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.metric-value {
    display: block;
    font-size: 1.5rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
}

.theory-section,
.interpretation-section,
.experiment-section {
    margin-top: 3rem;
}

.theory-section h2,
.interpretation-section h2,
.experiment-section h2 {
    font-size: 1.5rem;
    margin-bottom: 1.5rem;
    color: var(--neon-purple);
}

.theory-card {
    background: var(--bg-surface);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border-left: 3px solid var(--neon-blue);
}

.theory-card h3 {
    margin: 0 0 0.75rem;
    font-size: 1.1rem;
    color: var(--text-primary);
}

.theory-card p {
    margin: 0 0 0.75rem;
    color: var(--text-secondary);
    line-height: 1.6;
}

.theory-card ul {
    margin: 0.75rem 0 0;
    padding-left: 1.5rem;
    color: var(--text-secondary);
}

.theory-card li {
    margin-bottom: 0.5rem;
}

.interpretation-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
}

.interpretation-item {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    padding: 1rem;
    background: var(--bg-surface);
    border-radius: 8px;
}

.interpretation-icon {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 12px;
}

.interpretation-text {
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

.interpretation-text strong {
    color: var(--text-primary);
}

.experiment-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
}

.experiment-card {
    background: var(--bg-surface);
    border-radius: 12px;
    padding: 1.25rem;
    border: 1px solid var(--border-color);
    transition: border-color 0.3s, transform 0.3s;
}

.experiment-card:hover {
    border-color: var(--neon-cyan);
    transform: translateY(-2px);
}

.experiment-card h4 {
    margin: 0 0 0.75rem;
    font-size: 1rem;
    color: var(--neon-cyan);
}

.experiment-card p {
    margin: 0;
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

@media (max-width: 768px) {
    .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .metric-card.wide {
        grid-column: span 2;
    }

    .interpretation-grid,
    .experiment-grid {
        grid-template-columns: 1fr;
    }
}
</style>

<script src="../site/interactive/js/child_mind_viewer.js"></script>

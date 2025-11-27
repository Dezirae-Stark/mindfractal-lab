/**
 * UI Common — Shared UI Components and Mode Management
 * MindFractal Lab
 *
 * Provides mode toggle (Explorer/Researcher), theme management, and common UI utilities.
 */

// Mode state
const MindFractalUI = {
    mode: 'explorer', // 'explorer' or 'researcher'
    callbacks: [],

    /**
     * Initialize UI system
     */
    init() {
        this.loadMode();
        this.injectModeToggle();
        this.applyMode();
        console.log('MindFractal UI initialized, mode:', this.mode);
    },

    /**
     * Load mode from localStorage
     */
    loadMode() {
        const saved = localStorage.getItem('mindfractal-mode');
        if (saved === 'explorer' || saved === 'researcher') {
            this.mode = saved;
        }
    },

    /**
     * Save mode to localStorage
     */
    saveMode() {
        localStorage.setItem('mindfractal-mode', this.mode);
    },

    /**
     * Toggle between modes
     */
    toggleMode() {
        this.mode = this.mode === 'explorer' ? 'researcher' : 'explorer';
        this.saveMode();
        this.applyMode();
        this.notifyCallbacks();
    },

    /**
     * Set specific mode
     */
    setMode(mode) {
        if (mode === 'explorer' || mode === 'researcher') {
            this.mode = mode;
            this.saveMode();
            this.applyMode();
            this.notifyCallbacks();
        }
    },

    /**
     * Register callback for mode changes
     */
    onModeChange(callback) {
        this.callbacks.push(callback);
    },

    /**
     * Notify all callbacks
     */
    notifyCallbacks() {
        this.callbacks.forEach(cb => cb(this.mode));
    },

    /**
     * Apply mode to document
     */
    applyMode() {
        document.body.classList.remove('mode-explorer', 'mode-researcher');
        document.body.classList.add(`mode-${this.mode}`);

        // Update toggle button text
        const toggleBtn = document.getElementById('mode-toggle-btn');
        if (toggleBtn) {
            if (this.mode === 'explorer') {
                toggleBtn.innerHTML = '<span class="mode-icon">&#x1F50D;</span> Explorer Mode';
                toggleBtn.title = 'Switch to Researcher Mode (shows code)';
            } else {
                toggleBtn.innerHTML = '<span class="mode-icon">&#x1F4BB;</span> Researcher Mode';
                toggleBtn.title = 'Switch to Explorer Mode (simplified)';
            }
        }

        // Show/hide code blocks based on mode
        document.querySelectorAll('.researcher-only').forEach(el => {
            el.style.display = this.mode === 'researcher' ? '' : 'none';
        });

        document.querySelectorAll('.explorer-only').forEach(el => {
            el.style.display = this.mode === 'explorer' ? '' : 'none';
        });
    },

    /**
     * Inject mode toggle into page
     */
    injectModeToggle() {
        // Check if already injected
        if (document.getElementById('mode-toggle-container')) return;

        const container = document.createElement('div');
        container.id = 'mode-toggle-container';
        container.innerHTML = `
            <button id="mode-toggle-btn" class="mode-toggle-btn" aria-label="Toggle viewing mode">
                <span class="mode-icon">&#x1F50D;</span> Explorer Mode
            </button>
            <div class="mode-tooltip">
                <strong>Explorer Mode:</strong> Simplified interface for visual exploration<br>
                <strong>Researcher Mode:</strong> Full code access and technical details
            </div>
        `;

        document.body.appendChild(container);

        // Bind click event
        document.getElementById('mode-toggle-btn').addEventListener('click', () => {
            this.toggleMode();
        });
    }
};

/**
 * Loading overlay component
 */
const LoadingOverlay = {
    show(containerId, message = 'Computing...') {
        const container = document.getElementById(containerId);
        if (!container) return;

        let overlay = container.querySelector('.loading-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'loading-overlay';
            overlay.innerHTML = `
                <div class="loading-spinner"></div>
                <div class="loading-message">${message}</div>
            `;
            container.style.position = 'relative';
            container.appendChild(overlay);
        } else {
            overlay.querySelector('.loading-message').textContent = message;
            overlay.style.display = 'flex';
        }
    },

    hide(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const overlay = container.querySelector('.loading-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    },

    updateMessage(containerId, message) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const msgEl = container.querySelector('.loading-message');
        if (msgEl) {
            msgEl.textContent = message;
        }
    }
};

/**
 * Parameter slider factory
 */
function createSlider(config) {
    const {
        id,
        label,
        min = -2,
        max = 2,
        step = 0.1,
        value = 0,
        container,
        onChange
    } = config;

    const wrapper = document.createElement('div');
    wrapper.className = 'param-slider';
    wrapper.innerHTML = `
        <label for="${id}">
            ${label}: <span id="${id}-value">${value.toFixed(2)}</span>
        </label>
        <input type="range" id="${id}"
               min="${min}" max="${max}" step="${step}" value="${value}">
    `;

    if (container) {
        container.appendChild(wrapper);
    }

    const slider = wrapper.querySelector('input');
    const valueDisplay = wrapper.querySelector(`#${id}-value`);

    slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        valueDisplay.textContent = val.toFixed(2);
        if (onChange) onChange(val);
    });

    return {
        element: wrapper,
        slider,
        getValue: () => parseFloat(slider.value),
        setValue: (v) => {
            slider.value = v;
            valueDisplay.textContent = v.toFixed(2);
        }
    };
}

/**
 * Results panel component
 */
function createResultsPanel(containerId, title = 'Results') {
    const container = document.getElementById(containerId);
    if (!container) return null;

    const panel = document.createElement('div');
    panel.className = 'results-panel';
    panel.innerHTML = `
        <div class="results-header">${title}</div>
        <div class="results-content"></div>
    `;

    container.appendChild(panel);

    return {
        element: panel,
        setContent(html) {
            panel.querySelector('.results-content').innerHTML = html;
        },
        addItem(label, value) {
            const content = panel.querySelector('.results-content');
            const item = document.createElement('div');
            item.className = 'result-item';
            item.innerHTML = `<span class="result-label">${label}:</span> <span class="result-value">${value}</span>`;
            content.appendChild(item);
        },
        clear() {
            panel.querySelector('.results-content').innerHTML = '';
        }
    };
}

/**
 * Tab component for switching between views
 */
function createTabs(config) {
    const { containerId, tabs, onTabChange } = config;
    const container = document.getElementById(containerId);
    if (!container) return null;

    const tabBar = document.createElement('div');
    tabBar.className = 'tab-bar';

    const contentArea = document.createElement('div');
    contentArea.className = 'tab-content';

    tabs.forEach((tab, index) => {
        // Tab button
        const btn = document.createElement('button');
        btn.className = 'tab-btn' + (index === 0 ? ' active' : '');
        btn.textContent = tab.label;
        btn.dataset.tabId = tab.id;
        tabBar.appendChild(btn);

        // Content pane
        const pane = document.createElement('div');
        pane.className = 'tab-pane' + (index === 0 ? ' active' : '');
        pane.id = `tab-pane-${tab.id}`;
        if (tab.content) pane.innerHTML = tab.content;
        contentArea.appendChild(pane);
    });

    container.appendChild(tabBar);
    container.appendChild(contentArea);

    // Event handling
    tabBar.addEventListener('click', (e) => {
        if (!e.target.classList.contains('tab-btn')) return;

        const tabId = e.target.dataset.tabId;

        // Update active states
        tabBar.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        contentArea.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

        e.target.classList.add('active');
        document.getElementById(`tab-pane-${tabId}`).classList.add('active');

        if (onTabChange) onTabChange(tabId);
    });

    return {
        setActiveTab(tabId) {
            tabBar.querySelector(`[data-tab-id="${tabId}"]`).click();
        },
        getActiveTab() {
            return tabBar.querySelector('.tab-btn.active').dataset.tabId;
        },
        getPane(tabId) {
            return document.getElementById(`tab-pane-${tabId}`);
        }
    };
}

/**
 * Tooltip helper
 */
function showTooltip(element, text, position = 'top') {
    let tooltip = document.getElementById('global-tooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.id = 'global-tooltip';
        tooltip.className = 'global-tooltip';
        document.body.appendChild(tooltip);
    }

    tooltip.textContent = text;
    tooltip.style.display = 'block';

    const rect = element.getBoundingClientRect();
    const tipRect = tooltip.getBoundingClientRect();

    if (position === 'top') {
        tooltip.style.left = `${rect.left + rect.width/2 - tipRect.width/2}px`;
        tooltip.style.top = `${rect.top - tipRect.height - 8}px`;
    } else if (position === 'bottom') {
        tooltip.style.left = `${rect.left + rect.width/2 - tipRect.width/2}px`;
        tooltip.style.top = `${rect.bottom + 8}px`;
    }
}

function hideTooltip() {
    const tooltip = document.getElementById('global-tooltip');
    if (tooltip) tooltip.style.display = 'none';
}

/**
 * Format numbers for display
 */
function formatNumber(num, decimals = 3) {
    if (typeof num !== 'number' || isNaN(num)) return 'N/A';
    if (!isFinite(num)) return num > 0 ? '+Inf' : '-Inf';
    return num.toFixed(decimals);
}

/**
 * Debounce function for performance
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function for continuous updates
 */
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    MindFractalUI.init();
});

// Export
window.MindFractalUI = MindFractalUI;
window.LoadingOverlay = LoadingOverlay;
window.createSlider = createSlider;
window.createResultsPanel = createResultsPanel;
window.createTabs = createTabs;
window.showTooltip = showTooltip;
window.hideTooltip = hideTooltip;
window.formatNumber = formatNumber;
window.debounce = debounce;
window.throttle = throttle;

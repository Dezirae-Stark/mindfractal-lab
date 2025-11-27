/**
 * MindFractal Lab — Theme Switcher
 * Toggles between Cosmic Mode and Light Mode
 *
 * Features:
 * - localStorage persistence
 * - Particle sparkles toggle
 * - Nebula background toggle
 * - System preference detection
 */

(function() {
    'use strict';

    const STORAGE_KEY = 'mf-theme';
    const SPARKLES_KEY = 'mf-sparkles';
    const SPARKLE_COUNT = 30;

    let sparklesEnabled = true;
    let sparklesContainer = null;

    /**
     * Get the current theme from localStorage or system preference
     */
    function getStoredTheme() {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) return stored;

        // Check system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
            return 'light';
        }
        return 'cosmic';
    }

    /**
     * Get sparkles preference
     */
    function getSparklesPreference() {
        const stored = localStorage.getItem(SPARKLES_KEY);
        return stored !== 'false';
    }

    /**
     * Apply the theme to the document
     */
    function applyTheme(theme) {
        const body = document.body;

        if (theme === 'light') {
            body.classList.add('light-theme');
            body.classList.remove('cosmic-theme');
        } else {
            body.classList.remove('light-theme');
            body.classList.add('cosmic-theme');
        }

        localStorage.setItem(STORAGE_KEY, theme);

        // Dispatch custom event for other components to react
        window.dispatchEvent(new CustomEvent('mf-theme-change', { detail: { theme } }));
    }

    /**
     * Toggle between themes
     */
    function toggleTheme() {
        const currentTheme = document.body.classList.contains('light-theme') ? 'light' : 'cosmic';
        const newTheme = currentTheme === 'light' ? 'cosmic' : 'light';
        applyTheme(newTheme);
        return newTheme;
    }

    /**
     * Create the sparkles layer
     */
    function createSparklesLayer() {
        if (sparklesContainer) return;

        sparklesContainer = document.createElement('div');
        sparklesContainer.id = 'mf-sparkles-layer';
        document.body.appendChild(sparklesContainer);

        // Create sparkle elements
        for (let i = 0; i < SPARKLE_COUNT; i++) {
            createSparkle();
        }
    }

    /**
     * Create a single sparkle element
     */
    function createSparkle() {
        if (!sparklesContainer) return;

        const sparkle = document.createElement('div');
        sparkle.className = 'sparkle';

        // Random position
        sparkle.style.left = `${Math.random() * 100}%`;
        sparkle.style.top = `${Math.random() * 100 + 100}%`;

        // Random delay and duration
        sparkle.style.animationDelay = `${Math.random() * 8}s`;
        sparkle.style.animationDuration = `${6 + Math.random() * 4}s`;

        // Random color from neon palette
        const colors = [
            'rgba(107, 107, 255, 0.8)',
            'rgba(196, 90, 255, 0.8)',
            'rgba(57, 216, 255, 0.8)',
            'rgba(255, 79, 163, 0.8)',
            'rgba(255, 255, 255, 0.9)'
        ];
        sparkle.style.background = colors[Math.floor(Math.random() * colors.length)];
        sparkle.style.boxShadow = `0 0 6px ${sparkle.style.background}`;

        // Random size
        const size = 2 + Math.random() * 3;
        sparkle.style.width = `${size}px`;
        sparkle.style.height = `${size}px`;

        sparklesContainer.appendChild(sparkle);
    }

    /**
     * Remove the sparkles layer
     */
    function removeSparklesLayer() {
        if (sparklesContainer) {
            sparklesContainer.remove();
            sparklesContainer = null;
        }
    }

    /**
     * Toggle sparkles
     */
    function toggleSparkles() {
        sparklesEnabled = !sparklesEnabled;
        localStorage.setItem(SPARKLES_KEY, sparklesEnabled.toString());

        if (sparklesEnabled) {
            createSparklesLayer();
        } else {
            removeSparklesLayer();
        }

        return sparklesEnabled;
    }

    /**
     * Create the nebula background container
     */
    function createNebulaContainer() {
        if (document.getElementById('mf-nebula-bg')) return;

        const nebula = document.createElement('div');
        nebula.id = 'mf-nebula-bg';
        document.body.insertBefore(nebula, document.body.firstChild);
    }

    /**
     * Create the starfield background
     */
    function createStarfield() {
        if (document.querySelector('.starfield')) return;

        const starfield = document.createElement('div');
        starfield.className = 'starfield';
        document.body.insertBefore(starfield, document.body.firstChild);
    }

    /**
     * Create the theme switcher UI
     */
    function createThemeSwitcherUI() {
        // Check if already exists
        if (document.getElementById('mf-theme-switcher')) return;

        const container = document.createElement('div');
        container.id = 'mf-theme-switcher';

        // Theme toggle button
        const themeBtn = document.createElement('button');
        themeBtn.className = 'theme-toggle-btn';
        themeBtn.title = 'Toggle Theme';
        themeBtn.innerHTML = `
            <span class="icon-cosmic">🌌</span>
            <span class="icon-light">☀️</span>
        `;
        themeBtn.addEventListener('click', () => {
            toggleTheme();
        });

        // Sparkles toggle button
        const sparklesBtn = document.createElement('button');
        sparklesBtn.className = 'theme-toggle-btn sparkles-toggle-btn';
        sparklesBtn.title = 'Toggle Sparkles';
        sparklesBtn.innerHTML = '✨';
        if (sparklesEnabled) {
            sparklesBtn.classList.add('active');
        }
        sparklesBtn.addEventListener('click', () => {
            const enabled = toggleSparkles();
            sparklesBtn.classList.toggle('active', enabled);
        });

        container.appendChild(themeBtn);
        container.appendChild(sparklesBtn);
        document.body.appendChild(container);
    }

    /**
     * Initialize the theme system
     */
    function init() {
        // Apply stored theme immediately
        const theme = getStoredTheme();
        applyTheme(theme);

        // Get sparkles preference
        sparklesEnabled = getSparklesPreference();

        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', onDOMReady);
        } else {
            onDOMReady();
        }
    }

    /**
     * DOM ready handler
     */
    function onDOMReady() {
        // Create background elements
        createStarfield();
        createNebulaContainer();

        // Create sparkles if enabled
        if (sparklesEnabled) {
            createSparklesLayer();
        }

        // Create UI
        createThemeSwitcherUI();

        // Listen for system theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (e) => {
                // Only auto-switch if user hasn't set a preference
                if (!localStorage.getItem(STORAGE_KEY)) {
                    applyTheme(e.matches ? 'light' : 'cosmic');
                }
            });
        }
    }

    // Expose API
    window.MFTheme = {
        toggle: toggleTheme,
        apply: applyTheme,
        getCurrent: () => document.body.classList.contains('light-theme') ? 'light' : 'cosmic',
        toggleSparkles: toggleSparkles,
        isSparklesEnabled: () => sparklesEnabled
    };

    // Initialize
    init();

})();

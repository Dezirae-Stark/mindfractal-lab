/**
 * Pyodide Bootstrap — Initialize Pyodide for MindFractal Lab
 *
 * This module handles loading Pyodide and the custom Python modules
 * for interactive fractal visualization in the browser.
 */

// Global state
let pyodide = null;
let pyodideReady = false;
let loadingCallbacks = [];

// Configuration
const PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/';

/**
 * Initialize Pyodide and load required packages
 */
async function initPyodide() {
    if (pyodide) {
        return pyodide;
    }

    console.log('Loading Pyodide...');
    updateStatus('Loading Python runtime...');

    try {
        // Load Pyodide
        pyodide = await loadPyodide({
            indexURL: PYODIDE_CDN
        });

        updateStatus('Loading NumPy and Matplotlib...');

        // Load required packages
        await pyodide.loadPackage(['numpy', 'matplotlib']);

        updateStatus('Loading MindFractal modules...');

        // Load custom modules
        await loadCustomModules();

        pyodideReady = true;
        updateStatus('Ready!');

        // Call any waiting callbacks
        loadingCallbacks.forEach(cb => cb(pyodide));
        loadingCallbacks = [];

        return pyodide;
    } catch (error) {
        console.error('Failed to load Pyodide:', error);
        updateStatus('Error loading Python runtime');
        throw error;
    }
}

/**
 * Load custom Python modules from the server
 */
async function loadCustomModules() {
    const modules = [
        'fractal_core',
        'cy_core',
        'possibility_core'
    ];

    for (const moduleName of modules) {
        try {
            const response = await fetch(`./py/${moduleName}.py`);
            if (response.ok) {
                const code = await response.text();
                await pyodide.runPythonAsync(code);
                console.log(`Loaded module: ${moduleName}`);
            } else {
                console.warn(`Could not load module: ${moduleName}`);
            }
        } catch (error) {
            console.warn(`Error loading ${moduleName}:`, error);
        }
    }
}

/**
 * Update status display if element exists
 */
function updateStatus(message) {
    const statusEl = document.getElementById('pyodide-status');
    if (statusEl) {
        statusEl.textContent = message;
    }
    console.log('Pyodide status:', message);
}

/**
 * Wait for Pyodide to be ready
 */
function onPyodideReady(callback) {
    if (pyodideReady) {
        callback(pyodide);
    } else {
        loadingCallbacks.push(callback);
    }
}

/**
 * Run Python code and return result
 */
async function runPython(code) {
    if (!pyodide) {
        await initPyodide();
    }
    return await pyodide.runPythonAsync(code);
}

/**
 * Run Python function with arguments
 */
async function callPythonFunction(funcName, kwargs = {}) {
    if (!pyodide) {
        await initPyodide();
    }

    // Build Python call
    const argsStr = Object.entries(kwargs)
        .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
        .join(', ');

    const code = `${funcName}(${argsStr})`;
    return await pyodide.runPythonAsync(code);
}

/**
 * Compute and render basin of attraction
 */
async function computeBasin(c1, c2, resolution = 100) {
    const result = await runPython(`
        from fractal_core import compute_and_render_basin
        compute_and_render_basin(${c1}, ${c2}, ${resolution})
    `);
    return result;
}

/**
 * Compute and render Lyapunov map
 */
async function computeLyapunovMap(resolution = 50) {
    const result = await runPython(`
        from fractal_core import compute_and_render_lyapunov
        compute_and_render_lyapunov(${resolution})
    `);
    return result;
}

/**
 * Compute and render Mandelbrot set
 */
async function computeMandelbrot(resolution = 200) {
    const result = await runPython(`
        from cy_core import compute_and_render_mandelbrot
        compute_and_render_mandelbrot(${resolution})
    `);
    return result;
}

/**
 * Compute and render Julia set
 */
async function computeJulia(cRe, cIm, resolution = 200) {
    const result = await runPython(`
        from cy_core import compute_and_render_julia
        compute_and_render_julia(${cRe}, ${cIm}, ${resolution})
    `);
    return result;
}

/**
 * Compute and render CY slice
 */
async function computeCYSlice(resolution = 100, k = 2, eps = 0.5) {
    const result = await runPython(`
        from cy_core import compute_and_render_cy_slice
        compute_and_render_cy_slice(${resolution}, ${k}, ${eps})
    `);
    return result;
}

/**
 * Display base64 image in an element
 */
function displayImage(base64Data, elementId) {
    const img = document.getElementById(elementId);
    if (img) {
        img.src = 'data:image/png;base64,' + base64Data;
    }
}

/**
 * Create a loading indicator
 */
function showLoading(elementId) {
    const el = document.getElementById(elementId);
    if (el) {
        el.classList.add('loading');
        el.innerHTML = '<div class="loader">Computing...</div>';
    }
}

/**
 * Hide loading indicator
 */
function hideLoading(elementId) {
    const el = document.getElementById(elementId);
    if (el) {
        el.classList.remove('loading');
    }
}

// Export functions
window.MindFractal = {
    init: initPyodide,
    onReady: onPyodideReady,
    runPython,
    callPythonFunction,
    computeBasin,
    computeLyapunovMap,
    computeMandelbrot,
    computeJulia,
    computeCYSlice,
    displayImage,
    showLoading,
    hideLoading
};

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Check if we're on an interactive page
    if (document.querySelector('.interactive-demo')) {
        initPyodide().catch(console.error);
    }
});

/**
 * MindFractal Lab — Canvas Glow Shader
 * WebGL shader for glowing canvas border and energy pulse effects
 *
 * Features:
 * - Animated glow border
 * - Energy pulse effect
 * - Integrates with visualization canvases
 * - Performance-optimized
 */

(function() {
    'use strict';

    // Vertex shader
    const vertexShaderSource = `
        attribute vec2 a_position;
        attribute vec2 a_texCoord;
        varying vec2 v_texCoord;

        void main() {
            gl_Position = vec4(a_position, 0.0, 1.0);
            v_texCoord = a_texCoord;
        }
    `;

    // Fragment shader for glow effect
    const fragmentShaderSource = `
        precision mediump float;

        varying vec2 v_texCoord;
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform float u_intensity;
        uniform vec3 u_color1;
        uniform vec3 u_color2;

        void main() {
            vec2 uv = v_texCoord;
            vec2 center = vec2(0.5, 0.5);

            // Distance from edges
            float distX = min(uv.x, 1.0 - uv.x);
            float distY = min(uv.y, 1.0 - uv.y);
            float dist = min(distX, distY);

            // Create glow falloff
            float glowWidth = 0.15;
            float glow = 1.0 - smoothstep(0.0, glowWidth, dist);

            // Animated pulse
            float pulse = 0.5 + 0.5 * sin(u_time * 2.0);
            pulse = 0.7 + 0.3 * pulse;

            // Color gradient based on position
            float angle = atan(uv.y - 0.5, uv.x - 0.5);
            float colorMix = 0.5 + 0.5 * sin(angle * 2.0 + u_time);

            vec3 glowColor = mix(u_color1, u_color2, colorMix);

            // Corner intensity boost
            float cornerDist = length(uv - center);
            float cornerBoost = 1.0 + 0.3 * (1.0 - cornerDist);

            // Final glow
            float finalGlow = glow * pulse * u_intensity * cornerBoost;

            // Make center transparent
            float centerMask = smoothstep(0.0, glowWidth * 2.0, dist);

            gl_FragColor = vec4(glowColor, finalGlow * (1.0 - centerMask * 0.8));
        }
    `;

    class CanvasGlowShader {
        constructor(targetCanvas, options = {}) {
            this.targetCanvas = targetCanvas;
            if (!this.targetCanvas) return;

            this.options = {
                intensity: options.intensity || 0.8,
                color1: options.color1 || [0.42, 0.42, 1.0],  // #6b6bff
                color2: options.color2 || [0.77, 0.35, 1.0],  // #c45aff
                ...options
            };

            this.canvas = null;
            this.gl = null;
            this.program = null;
            this.animationId = null;
            this.startTime = Date.now();

            // Check WebGL support
            if (!this.isWebGLSupported()) {
                this.useCSSFallback();
                return;
            }

            this.init();
        }

        isWebGLSupported() {
            try {
                const canvas = document.createElement('canvas');
                return !!(window.WebGLRenderingContext &&
                    (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
            } catch (e) {
                return false;
            }
        }

        useCSSFallback() {
            // CSS-based glow effect
            this.targetCanvas.style.boxShadow = `
                0 0 20px rgba(107, 107, 255, 0.3),
                0 0 40px rgba(196, 90, 255, 0.2),
                0 0 60px rgba(107, 107, 255, 0.1)
            `;
        }

        init() {
            const parent = this.targetCanvas.parentElement;
            if (!parent) return;

            // Create overlay canvas
            this.canvas = document.createElement('canvas');
            this.canvas.className = 'canvas-glow-overlay';
            this.canvas.style.position = 'absolute';
            this.canvas.style.top = '0';
            this.canvas.style.left = '0';
            this.canvas.style.width = '100%';
            this.canvas.style.height = '100%';
            this.canvas.style.pointerEvents = 'none';
            this.canvas.style.zIndex = '10';
            this.canvas.style.mixBlendMode = 'screen';

            // Make parent relative if not already
            const parentPosition = window.getComputedStyle(parent).position;
            if (parentPosition === 'static') {
                parent.style.position = 'relative';
            }

            parent.appendChild(this.canvas);

            // Get WebGL context
            this.gl = this.canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false }) ||
                      this.canvas.getContext('experimental-webgl', { alpha: true, premultipliedAlpha: false });

            if (!this.gl) {
                this.useCSSFallback();
                return;
            }

            // Enable blending
            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

            // Compile shaders
            const vertexShader = this.compileShader(this.gl.VERTEX_SHADER, vertexShaderSource);
            const fragmentShader = this.compileShader(this.gl.FRAGMENT_SHADER, fragmentShaderSource);

            if (!vertexShader || !fragmentShader) {
                this.useCSSFallback();
                return;
            }

            // Create and link program
            this.program = this.gl.createProgram();
            this.gl.attachShader(this.program, vertexShader);
            this.gl.attachShader(this.program, fragmentShader);
            this.gl.linkProgram(this.program);

            if (!this.gl.getProgramParameter(this.program, this.gl.LINK_STATUS)) {
                console.error('Program link error:', this.gl.getProgramInfoLog(this.program));
                this.useCSSFallback();
                return;
            }

            // Create geometry
            this.setupGeometry();

            // Get uniform locations
            this.uniforms = {
                time: this.gl.getUniformLocation(this.program, 'u_time'),
                resolution: this.gl.getUniformLocation(this.program, 'u_resolution'),
                intensity: this.gl.getUniformLocation(this.program, 'u_intensity'),
                color1: this.gl.getUniformLocation(this.program, 'u_color1'),
                color2: this.gl.getUniformLocation(this.program, 'u_color2')
            };

            // Handle resize
            this.resize();
            window.addEventListener('resize', () => this.resize());

            // Start animation
            this.animate();

            // Listen for theme changes
            window.addEventListener('mf-theme-change', (e) => {
                if (e.detail.theme === 'light') {
                    this.pause();
                } else {
                    this.resume();
                }
            });

            // Check initial theme
            if (document.body.classList.contains('light-theme')) {
                this.pause();
            }
        }

        compileShader(type, source) {
            const shader = this.gl.createShader(type);
            this.gl.shaderSource(shader, source);
            this.gl.compileShader(shader);

            if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
                console.error('Shader compile error:', this.gl.getShaderInfoLog(shader));
                this.gl.deleteShader(shader);
                return null;
            }

            return shader;
        }

        setupGeometry() {
            // Positions
            const positions = new Float32Array([
                -1, -1,
                 1, -1,
                -1,  1,
                 1,  1
            ]);

            const positionBuffer = this.gl.createBuffer();
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
            this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

            this.positionLocation = this.gl.getAttribLocation(this.program, 'a_position');

            // Texture coordinates
            const texCoords = new Float32Array([
                0, 0,
                1, 0,
                0, 1,
                1, 1
            ]);

            const texCoordBuffer = this.gl.createBuffer();
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, texCoordBuffer);
            this.gl.bufferData(this.gl.ARRAY_BUFFER, texCoords, this.gl.STATIC_DRAW);

            this.texCoordLocation = this.gl.getAttribLocation(this.program, 'a_texCoord');

            this.positionBuffer = positionBuffer;
            this.texCoordBuffer = texCoordBuffer;
        }

        resize() {
            if (!this.canvas || !this.targetCanvas) return;

            const rect = this.targetCanvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;

            const width = Math.floor(rect.width * dpr);
            const height = Math.floor(rect.height * dpr);

            if (this.canvas.width !== width || this.canvas.height !== height) {
                this.canvas.width = width;
                this.canvas.height = height;
                if (this.gl) {
                    this.gl.viewport(0, 0, width, height);
                }
            }
        }

        animate() {
            if (!this.gl || !this.program) return;

            const time = (Date.now() - this.startTime) / 1000;

            this.gl.clearColor(0, 0, 0, 0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);

            this.gl.useProgram(this.program);

            // Set uniforms
            this.gl.uniform1f(this.uniforms.time, time);
            this.gl.uniform2f(this.uniforms.resolution, this.canvas.width, this.canvas.height);
            this.gl.uniform1f(this.uniforms.intensity, this.options.intensity);
            this.gl.uniform3fv(this.uniforms.color1, this.options.color1);
            this.gl.uniform3fv(this.uniforms.color2, this.options.color2);

            // Set up position attribute
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
            this.gl.enableVertexAttribArray(this.positionLocation);
            this.gl.vertexAttribPointer(this.positionLocation, 2, this.gl.FLOAT, false, 0, 0);

            // Set up texCoord attribute
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texCoordBuffer);
            this.gl.enableVertexAttribArray(this.texCoordLocation);
            this.gl.vertexAttribPointer(this.texCoordLocation, 2, this.gl.FLOAT, false, 0, 0);

            // Draw
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

            this.animationId = requestAnimationFrame(() => this.animate());
        }

        pause() {
            if (this.animationId) {
                cancelAnimationFrame(this.animationId);
                this.animationId = null;
            }
            if (this.canvas) {
                this.canvas.style.opacity = '0';
            }
        }

        resume() {
            if (this.canvas) {
                this.canvas.style.opacity = '1';
            }
            if (!this.animationId) {
                this.animate();
            }
        }

        setIntensity(value) {
            this.options.intensity = Math.max(0, Math.min(1, value));
        }

        setColors(color1, color2) {
            if (color1) this.options.color1 = color1;
            if (color2) this.options.color2 = color2;
        }

        destroy() {
            this.pause();
            if (this.canvas && this.canvas.parentNode) {
                this.canvas.parentNode.removeChild(this.canvas);
            }
        }
    }

    /**
     * Auto-apply glow to visualization canvases
     */
    function initCanvasGlows() {
        // Find all canvas elements in mf-canvas-panel containers
        const canvasPanels = document.querySelectorAll('.mf-canvas-panel canvas');

        canvasPanels.forEach(canvas => {
            // Skip if already has glow
            if (canvas.dataset.hasGlow) return;

            new CanvasGlowShader(canvas);
            canvas.dataset.hasGlow = 'true';
        });
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initCanvasGlows);
    } else {
        setTimeout(initCanvasGlows, 500);
    }

    // Also watch for dynamically added canvases
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === 1) {
                    const canvases = node.querySelectorAll ? node.querySelectorAll('.mf-canvas-panel canvas') : [];
                    canvases.forEach(canvas => {
                        if (!canvas.dataset.hasGlow) {
                            setTimeout(() => {
                                new CanvasGlowShader(canvas);
                                canvas.dataset.hasGlow = 'true';
                            }, 100);
                        }
                    });
                }
            });
        });
    });

    if (document.body) {
        observer.observe(document.body, { childList: true, subtree: true });
    }

    // Expose class
    window.MFCanvasGlowShader = CanvasGlowShader;

})();

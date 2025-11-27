/**
 * MindFractal Lab — Nebula Background Shader
 * WebGL fragment shader for animated cosmic nebula effect
 *
 * Features:
 * - Perlin-like noise for organic movement
 * - Color-shifting based on time
 * - Mobile-safe with static fallback
 * - Adjustable opacity and speed
 */

(function() {
    'use strict';

    // Vertex shader (simple passthrough)
    const vertexShaderSource = `
        attribute vec2 a_position;
        void main() {
            gl_Position = vec4(a_position, 0.0, 1.0);
        }
    `;

    // Fragment shader with Perlin noise-inspired nebula
    const fragmentShaderSource = `
        precision mediump float;

        uniform float u_time;
        uniform vec2 u_resolution;
        uniform float u_opacity;
        uniform float u_speed;

        // Pseudo-random function
        float random(vec2 st) {
            return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
        }

        // 2D noise function
        float noise(vec2 st) {
            vec2 i = floor(st);
            vec2 f = fract(st);

            float a = random(i);
            float b = random(i + vec2(1.0, 0.0));
            float c = random(i + vec2(0.0, 1.0));
            float d = random(i + vec2(1.0, 1.0));

            vec2 u = f * f * (3.0 - 2.0 * f);

            return mix(a, b, u.x) +
                   (c - a) * u.y * (1.0 - u.x) +
                   (d - b) * u.x * u.y;
        }

        // Fractal Brownian Motion
        float fbm(vec2 st) {
            float value = 0.0;
            float amplitude = 0.5;
            float frequency = 1.0;

            for (int i = 0; i < 5; i++) {
                value += amplitude * noise(st * frequency);
                st *= 2.0;
                amplitude *= 0.5;
            }
            return value;
        }

        void main() {
            vec2 st = gl_FragCoord.xy / u_resolution.xy;

            // Adjust aspect ratio
            st.x *= u_resolution.x / u_resolution.y;

            // Slow time for smooth animation
            float t = u_time * u_speed * 0.05;

            // Create multiple noise layers for depth
            float n1 = fbm(st * 2.0 + vec2(t * 0.3, t * 0.2));
            float n2 = fbm(st * 3.0 - vec2(t * 0.2, t * 0.4));
            float n3 = fbm(st * 4.0 + vec2(t * 0.1, -t * 0.1));

            // Color palette - cosmic neon
            vec3 color1 = vec3(0.42, 0.42, 1.0);   // #6b6bff - neon purple
            vec3 color2 = vec3(0.77, 0.35, 1.0);   // #c45aff - neon magenta
            vec3 color3 = vec3(0.22, 0.85, 1.0);   // #39d8ff - neon cyan
            vec3 color4 = vec3(1.0, 0.31, 0.64);   // #ff4fa3 - neon pink

            // Mix colors based on noise
            vec3 nebula = mix(color1, color2, n1);
            nebula = mix(nebula, color3, n2 * 0.5);
            nebula = mix(nebula, color4, n3 * 0.3);

            // Add brightness variation
            float brightness = 0.3 + 0.2 * sin(t + n1 * 3.0);

            // Darken edges for vignette effect
            float vignette = 1.0 - length(st - vec2(0.5 * u_resolution.x / u_resolution.y, 0.5)) * 0.5;
            vignette = clamp(vignette, 0.0, 1.0);

            // Final color
            vec3 finalColor = nebula * brightness * vignette;

            // Add subtle stars
            float starNoise = random(st * 100.0 + t * 0.01);
            if (starNoise > 0.995) {
                finalColor += vec3(0.8);
            }

            gl_FragColor = vec4(finalColor, u_opacity);
        }
    `;

    class NebulaShader {
        constructor(containerId) {
            this.container = document.getElementById(containerId);
            if (!this.container) {
                console.warn('Nebula shader container not found:', containerId);
                return;
            }

            this.canvas = null;
            this.gl = null;
            this.program = null;
            this.animationId = null;
            this.startTime = Date.now();
            this.opacity = 0.6;
            this.speed = 1.0;

            // Check for WebGL support
            if (!this.isWebGLSupported()) {
                this.useFallback();
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

        useFallback() {
            // Use CSS gradient as fallback
            this.container.style.background = `
                linear-gradient(135deg,
                    rgba(107, 107, 255, 0.3) 0%,
                    rgba(196, 90, 255, 0.2) 30%,
                    rgba(13, 14, 18, 0.9) 50%,
                    rgba(57, 216, 255, 0.2) 70%,
                    rgba(255, 79, 163, 0.3) 100%)
            `;
        }

        init() {
            // Create canvas
            this.canvas = document.createElement('canvas');
            this.canvas.style.width = '100%';
            this.canvas.style.height = '100%';
            this.canvas.style.display = 'block';
            this.container.appendChild(this.canvas);

            // Get WebGL context
            this.gl = this.canvas.getContext('webgl') ||
                      this.canvas.getContext('experimental-webgl');

            if (!this.gl) {
                this.useFallback();
                return;
            }

            // Compile shaders
            const vertexShader = this.compileShader(this.gl.VERTEX_SHADER, vertexShaderSource);
            const fragmentShader = this.compileShader(this.gl.FRAGMENT_SHADER, fragmentShaderSource);

            if (!vertexShader || !fragmentShader) {
                this.useFallback();
                return;
            }

            // Create program
            this.program = this.gl.createProgram();
            this.gl.attachShader(this.program, vertexShader);
            this.gl.attachShader(this.program, fragmentShader);
            this.gl.linkProgram(this.program);

            if (!this.gl.getProgramParameter(this.program, this.gl.LINK_STATUS)) {
                console.error('Program link error:', this.gl.getProgramInfoLog(this.program));
                this.useFallback();
                return;
            }

            // Create geometry (full-screen quad)
            const positions = new Float32Array([
                -1, -1,
                 1, -1,
                -1,  1,
                 1,  1
            ]);

            const buffer = this.gl.createBuffer();
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
            this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

            // Get attribute/uniform locations
            this.positionLocation = this.gl.getAttribLocation(this.program, 'a_position');
            this.timeLocation = this.gl.getUniformLocation(this.program, 'u_time');
            this.resolutionLocation = this.gl.getUniformLocation(this.program, 'u_resolution');
            this.opacityLocation = this.gl.getUniformLocation(this.program, 'u_opacity');
            this.speedLocation = this.gl.getUniformLocation(this.program, 'u_speed');

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

        resize() {
            if (!this.canvas) return;

            const displayWidth = this.container.clientWidth;
            const displayHeight = this.container.clientHeight;

            if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight) {
                this.canvas.width = displayWidth;
                this.canvas.height = displayHeight;
                if (this.gl) {
                    this.gl.viewport(0, 0, displayWidth, displayHeight);
                }
            }
        }

        animate() {
            if (!this.gl || !this.program) return;

            const time = (Date.now() - this.startTime) / 1000;

            this.gl.useProgram(this.program);

            // Set uniforms
            this.gl.uniform1f(this.timeLocation, time);
            this.gl.uniform2f(this.resolutionLocation, this.canvas.width, this.canvas.height);
            this.gl.uniform1f(this.opacityLocation, this.opacity);
            this.gl.uniform1f(this.speedLocation, this.speed);

            // Set up attribute
            this.gl.enableVertexAttribArray(this.positionLocation);
            this.gl.vertexAttribPointer(this.positionLocation, 2, this.gl.FLOAT, false, 0, 0);

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

        setOpacity(value) {
            this.opacity = Math.max(0, Math.min(1, value));
        }

        setSpeed(value) {
            this.speed = Math.max(0.1, Math.min(3, value));
        }

        destroy() {
            this.pause();
            if (this.canvas && this.canvas.parentNode) {
                this.canvas.parentNode.removeChild(this.canvas);
            }
        }
    }

    // Auto-initialize when nebula container exists
    function init() {
        const container = document.getElementById('mf-nebula-bg');
        if (container) {
            window.mfNebula = new NebulaShader('mf-nebula-bg');
        }
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        // Wait a bit for theme switcher to create container
        setTimeout(init, 100);
    }

    // Expose class
    window.MFNebulaShader = NebulaShader;

})();

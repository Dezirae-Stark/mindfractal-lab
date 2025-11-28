/**
 * Cytherea 3D Avatar Controller
 * Three.js implementation for high-end devices
 */

class Cytherea3DAvatar {
  constructor() {
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.model = null;
    this.animationMixer = null;
    this.isInitialized = false;
    this.currentMood = 'neutral';
    this.container = null;
    
    // Animation clips for different moods
    this.moodAnimations = {};
    
    // Particle systems for each mood
    this.particleSystems = {};
  }

  async init(containerId) {
    console.log('[Cytherea 3D] Initializing Three.js avatar system...');
    
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error('[Cytherea 3D] Container not found:', containerId);
      return false;
    }

    try {
      // Dynamically load Three.js
      await this.loadThreeJS();
      
      // Initialize scene
      this.initScene();
      
      // Load avatar model
      await this.loadAvatarModel();
      
      // Setup animations
      this.setupAnimations();
      
      // Add lighting
      this.setupLighting();
      
      // Add particle effects
      this.setupParticleEffects();
      
      // Start render loop
      this.startRenderLoop();
      
      // Handle resize
      window.addEventListener('resize', () => this.onResize());
      
      this.isInitialized = true;
      console.log('[Cytherea 3D] Initialization complete');
      return true;
      
    } catch (error) {
      console.error('[Cytherea 3D] Initialization failed:', error);
      return false;
    }
  }

  async loadThreeJS() {
    // Load Three.js from CDN
    return new Promise((resolve, reject) => {
      if (window.THREE) {
        resolve();
        return;
      }
      
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r147/three.min.js';
      script.onload = async () => {
        // Also load GLTF loader
        const gltfScript = document.createElement('script');
        gltfScript.src = 'https://cdn.jsdelivr.net/npm/three@0.147.0/examples/js/loaders/GLTFLoader.js';
        gltfScript.onload = resolve;
        gltfScript.onerror = reject;
        document.head.appendChild(gltfScript);
      };
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  initScene() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    
    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f1729);
    
    // Camera
    this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    this.camera.position.set(0, 1.6, 3);
    this.camera.lookAt(0, 1.6, 0);
    
    // Renderer
    this.renderer = new THREE.WebGLRenderer({ 
      antialias: true, 
      alpha: true,
      powerPreference: "high-performance"
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    
    this.container.appendChild(this.renderer.domElement);
  }

  async loadAvatarModel() {
    // For now, create a stylized geometric avatar
    // In production, you'd load a real 3D model from photos
    await this.createGeometricAvatar();
  }

  async createGeometricAvatar() {
    const avatarGroup = new THREE.Group();
    
    // Head (main sphere)
    const headGeometry = new THREE.SphereGeometry(0.8, 32, 32);
    const headMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xf4d5c3,
      roughness: 0.1,
      metalness: 0.0,
      clearcoat: 0.3,
      clearcoatRoughness: 0.1
    });
    const head = new THREE.Mesh(headGeometry, headMaterial);
    head.position.y = 1.6;
    head.castShadow = true;
    avatarGroup.add(head);
    
    // Eyes
    const eyeGeometry = new THREE.SphereGeometry(0.15, 16, 16);
    const eyeMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xff6ec7, // Purple iris
      roughness: 0.0,
      metalness: 0.0,
      transmission: 0.9,
      thickness: 0.1
    });
    
    const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    leftEye.position.set(-0.25, 1.7, 0.6);
    avatarGroup.add(leftEye);
    
    const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    rightEye.position.set(0.25, 1.7, 0.6);
    avatarGroup.add(rightEye);
    
    // Pupils with glow
    const pupilGeometry = new THREE.SphereGeometry(0.05, 16, 16);
    const pupilMaterial = new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.8
    });
    
    const leftPupil = new THREE.Mesh(pupilGeometry, pupilMaterial);
    leftPupil.position.set(-0.25, 1.7, 0.7);
    avatarGroup.add(leftPupil);
    
    const rightPupil = new THREE.Mesh(pupilGeometry, pupilMaterial);
    rightPupil.position.set(0.25, 1.7, 0.7);
    avatarGroup.add(rightPupil);
    
    // Hair (crystalline structures)
    this.createHairGeometry(avatarGroup);
    
    this.model = avatarGroup;
    this.scene.add(avatarGroup);
    
    // Store references for animation
    this.eyeLeft = leftEye;
    this.eyeRight = rightEye;
    this.head = head;
  }

  createHairGeometry(parent) {
    const hairGroup = new THREE.Group();
    
    // Create crystalline hair strands
    for (let i = 0; i < 12; i++) {
      const angle = (i / 12) * Math.PI * 2;
      const radius = 0.9;
      
      const strandGeometry = new THREE.CylinderGeometry(0.05, 0.02, 0.8, 8);
      const strandMaterial = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color().setHSL(0.8, 0.6, 0.4), // Purple
        roughness: 0.1,
        metalness: 0.2,
        transparent: true,
        opacity: 0.9
      });
      
      const strand = new THREE.Mesh(strandGeometry, strandMaterial);
      strand.position.x = Math.cos(angle) * radius;
      strand.position.y = 2.0;
      strand.position.z = Math.sin(angle) * radius;
      strand.rotation.z = angle + Math.PI / 2;
      strand.rotation.y = Math.random() * 0.3;
      
      hairGroup.add(strand);
    }
    
    parent.add(hairGroup);
    this.hair = hairGroup;
  }

  setupAnimations() {
    // Create basic breathing animation
    const breathingKeyframes = [];
    const times = [0, 1, 2];
    const values = [
      0, 0, 0,    // Start position
      0, 0.02, 0, // Up
      0, 0, 0     // Back to start
    ];
    
    const breathingTrack = new THREE.VectorKeyframeTrack(
      '.position',
      times,
      values
    );
    
    const breathingClip = new THREE.AnimationClip('breathing', 2, [breathingTrack]);
    
    this.animationMixer = new THREE.AnimationMixer(this.model);
    const breathingAction = this.animationMixer.clipAction(breathingClip);
    breathingAction.loop = THREE.LoopRepeat;
    breathingAction.play();
  }

  setupLighting() {
    // Key light
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
    keyLight.position.set(2, 4, 3);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    this.scene.add(keyLight);
    
    // Fill light
    const fillLight = new THREE.DirectionalLight(0x9361ed, 0.6);
    fillLight.position.set(-2, 2, 1);
    this.scene.add(fillLight);
    
    // Rim light
    const rimLight = new THREE.DirectionalLight(0xff4fa3, 0.8);
    rimLight.position.set(0, 2, -3);
    this.scene.add(rimLight);
    
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0x4a2970, 0.3);
    this.scene.add(ambientLight);
    
    // Dynamic mood lighting
    this.moodLights = {
      key: keyLight,
      fill: fillLight,
      rim: rimLight
    };
  }

  setupParticleEffects() {
    // Create particle systems for each mood
    this.createMoodParticles();
  }

  createMoodParticles() {
    const particleCount = 50;
    
    // Shared geometry
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const velocities = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 4;
      positions[i * 3 + 1] = Math.random() * 3 + 1;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 4;
      
      velocities[i * 3] = (Math.random() - 0.5) * 0.02;
      velocities[i * 3 + 1] = Math.random() * 0.01;
      velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.02;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
    
    // Different materials for different moods
    const materials = {
      neutral: new THREE.PointsMaterial({
        color: 0x7c3aed,
        size: 0.02,
        transparent: true,
        opacity: 0.6
      }),
      focused: new THREE.PointsMaterial({
        color: 0xff4fa3,
        size: 0.03,
        transparent: true,
        opacity: 0.8
      }),
      dream: new THREE.PointsMaterial({
        color: 0x6ee7b7,
        size: 0.04,
        transparent: true,
        opacity: 0.5
      }),
      overload: new THREE.PointsMaterial({
        color: 0x9ca3af,
        size: 0.015,
        transparent: true,
        opacity: 0.9
      }),
      celebrate: new THREE.PointsMaterial({
        color: 0xfde047,
        size: 0.05,
        transparent: true,
        opacity: 0.9
      })
    };
    
    // Create particle systems
    Object.keys(materials).forEach(mood => {
      const particles = new THREE.Points(geometry.clone(), materials[mood]);
      particles.visible = mood === 'neutral';
      this.scene.add(particles);
      this.particleSystems[mood] = particles;
    });
  }

  setMood(mood) {
    if (!this.isInitialized || this.currentMood === mood) return;
    
    console.log(`[Cytherea 3D] Changing mood to: ${mood}`);
    this.currentMood = mood;
    
    // Update lighting
    this.updateMoodLighting(mood);
    
    // Update particles
    this.updateMoodParticles(mood);
    
    // Update avatar animations
    this.updateMoodAnimation(mood);
    
    // Dispatch event for integration
    document.dispatchEvent(new CustomEvent('cytherea-3d-mood-changed', {
      detail: { mood }
    }));
  }

  updateMoodLighting(mood) {
    const lightConfigs = {
      neutral: { key: 0xffffff, fill: 0x9361ed, rim: 0xff4fa3 },
      focused: { key: 0xffffff, fill: 0xff4fa3, rim: 0xff6ec7 },
      dream: { key: 0xe0f2fe, fill: 0x6ee7b7, rim: 0x9361ed },
      overload: { key: 0xf3f4f6, fill: 0x9ca3af, rim: 0xef4444 },
      celebrate: { key: 0xfef3c7, fill: 0xfde047, rim: 0xff4fa3 }
    };
    
    const config = lightConfigs[mood] || lightConfigs.neutral;
    
    this.moodLights.key.color.setHex(config.key);
    this.moodLights.fill.color.setHex(config.fill);
    this.moodLights.rim.color.setHex(config.rim);
  }

  updateMoodParticles(mood) {
    Object.keys(this.particleSystems).forEach(m => {
      this.particleSystems[m].visible = m === mood;
    });
  }

  updateMoodAnimation(mood) {
    // Add mood-specific animations
    switch (mood) {
      case 'focused':
        this.addFocusedAnimation();
        break;
      case 'dream':
        this.addDreamAnimation();
        break;
      case 'overload':
        this.addOverloadAnimation();
        break;
      case 'celebrate':
        this.addCelebrateAnimation();
        break;
      default:
        this.resetToNeutralAnimation();
    }
  }

  addFocusedAnimation() {
    // Subtle head tracking movement
    if (this.head) {
      const focusAnimation = () => {
        const time = Date.now() * 0.001;
        this.head.rotation.y = Math.sin(time * 0.5) * 0.1;
        this.head.rotation.x = Math.cos(time * 0.3) * 0.05;
      };
      this.currentAnimation = focusAnimation;
    }
  }

  addDreamAnimation() {
    // Floating motion
    if (this.model) {
      const dreamAnimation = () => {
        const time = Date.now() * 0.001;
        this.model.position.y = Math.sin(time * 0.8) * 0.1;
        this.model.rotation.z = Math.sin(time * 0.6) * 0.02;
      };
      this.currentAnimation = dreamAnimation;
    }
  }

  addOverloadAnimation() {
    // Glitch-like movement
    if (this.head) {
      const overloadAnimation = () => {
        if (Math.random() < 0.1) {
          this.head.position.x = (Math.random() - 0.5) * 0.05;
          this.head.position.y = 1.6 + (Math.random() - 0.5) * 0.05;
        } else {
          this.head.position.x = 0;
          this.head.position.y = 1.6;
        }
      };
      this.currentAnimation = overloadAnimation;
    }
  }

  addCelebrateAnimation() {
    // Bouncy celebration
    if (this.model) {
      const celebrateAnimation = () => {
        const time = Date.now() * 0.001;
        this.model.position.y = Math.abs(Math.sin(time * 2)) * 0.15;
        this.model.rotation.y = Math.sin(time * 1.5) * 0.1;
      };
      this.currentAnimation = celebrateAnimation;
    }
  }

  resetToNeutralAnimation() {
    this.currentAnimation = null;
    if (this.model) {
      this.model.position.set(0, 0, 0);
      this.model.rotation.set(0, 0, 0);
    }
    if (this.head) {
      this.head.position.set(0, 1.6, 0);
      this.head.rotation.set(0, 0, 0);
    }
  }

  startRenderLoop() {
    const animate = () => {
      requestAnimationFrame(animate);
      
      if (this.animationMixer) {
        this.animationMixer.update(0.016);
      }
      
      // Update custom mood animations
      if (this.currentAnimation) {
        this.currentAnimation();
      }
      
      // Update particles
      this.updateParticles();
      
      this.renderer.render(this.scene, this.camera);
    };
    
    animate();
  }

  updateParticles() {
    Object.values(this.particleSystems).forEach(system => {
      if (system.visible) {
        const positions = system.geometry.attributes.position.array;
        const velocities = system.geometry.attributes.velocity.array;
        
        for (let i = 0; i < positions.length; i += 3) {
          positions[i] += velocities[i];
          positions[i + 1] += velocities[i + 1];
          positions[i + 2] += velocities[i + 2];
          
          // Reset particles that go too far
          if (positions[i + 1] > 4) {
            positions[i + 1] = 0;
          }
        }
        
        system.geometry.attributes.position.needsUpdate = true;
      }
    });
  }

  onResize() {
    if (!this.camera || !this.renderer) return;
    
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  cleanup() {
    if (this.renderer) {
      this.container.removeChild(this.renderer.domElement);
    }
    
    if (this.scene) {
      this.scene.clear();
    }
    
    this.isInitialized = false;
    console.log('[Cytherea 3D] Cleaned up');
  }
}

// Integration with existing avatar system
(function() {
  let cytherea3D = null;
  
  const toggleContainer = document.getElementById('cy-3d-toggle');
  const toggle = document.getElementById('cy-enable-3d');
  
  if (!toggleContainer || !toggle) return;
  
  // Check device capabilities
  if (!CythereaDeviceProfile.shouldOffer3D()) {
    toggleContainer.hidden = true;
    return;
  }
  
  // Show toggle on capable devices
  toggleContainer.hidden = false;
  
  toggle.addEventListener('change', async (event) => {
    const enabled = event.target.checked;
    const avatarContainer = document.getElementById('cy-avatar-realistic');
    
    if (enabled && !cytherea3D) {
      // Hide 2D avatar
      if (avatarContainer) {
        avatarContainer.style.display = 'none';
      }
      
      // Create 3D container
      const container3D = document.createElement('div');
      container3D.id = 'cy-avatar-3d';
      container3D.style.cssText = `
        width: 100%;
        height: 400px;
        border-radius: 1.25rem;
        overflow: hidden;
        background: radial-gradient(circle at top, #1b1638, #050510);
      `;
      
      avatarContainer.parentNode.insertBefore(container3D, avatarContainer);
      
      // Initialize 3D avatar
      cytherea3D = new Cytherea3DAvatar();
      const success = await cytherea3D.init('cy-avatar-3d');
      
      if (success) {
        // Listen for mood changes from 2D system
        document.addEventListener('cytherea-mood-changed', (event) => {
          if (cytherea3D) {
            cytherea3D.setMood(event.detail.mood);
          }
        });
        
        console.log('[Cytherea 3D] 3D avatar activated');
      } else {
        // Fallback to 2D
        toggle.checked = false;
        container3D.remove();
        if (avatarContainer) {
          avatarContainer.style.display = '';
        }
      }
      
    } else if (!enabled && cytherea3D) {
      // Switch back to 2D
      const container3D = document.getElementById('cy-avatar-3d');
      if (container3D) {
        container3D.remove();
      }
      
      if (avatarContainer) {
        avatarContainer.style.display = '';
      }
      
      cytherea3D.cleanup();
      cytherea3D = null;
      console.log('[Cytherea 3D] Switched back to 2D avatar');
    }
  });
})();
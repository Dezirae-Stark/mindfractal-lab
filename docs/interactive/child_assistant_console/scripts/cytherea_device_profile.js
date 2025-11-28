/**
 * Cytherea Device Profile
 * Detects device capabilities for optimal avatar experience
 */

const CythereaDeviceProfile = (function () {
  function hasWebGL() {
    try {
      const canvas = document.createElement('canvas');
      return !!(
        window.WebGLRenderingContext &&
        (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
      );
    } catch (e) {
      return false;
    }
  }

  function isMobile() {
    return /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent || '');
  }

  function isLowPowerMode() {
    // Check for reduced motion preference as a proxy for low power mode
    return window.matchMedia && 
           window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function hasGoodNetworkSpeed() {
    // Use Network Information API if available
    if ('connection' in navigator) {
      const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
      if (connection) {
        // Consider 3g and above as good
        return connection.effectiveType === '4g' || connection.effectiveType === '3g';
      }
    }
    return true; // Assume good if we can't detect
  }

  function getDeviceMemory() {
    // Device Memory API (Chrome)
    if ('deviceMemory' in navigator) {
      return navigator.deviceMemory;
    }
    return 4; // Default assumption
  }

  function shouldOffer3D() {
    // Only offer 3D on desktop with WebGL, good network, and sufficient memory
    return hasWebGL() && 
           !isMobile() && 
           !isLowPowerMode() && 
           hasGoodNetworkSpeed() &&
           getDeviceMemory() >= 4;
  }

  function getOptimalImageFormat() {
    // Check WebP support
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 1;
    const dataUrl = canvas.toDataURL('image/webp');
    const hasWebP = dataUrl.indexOf('image/webp') === 5;
    
    return hasWebP ? 'webp' : 'jpg';
  }

  function getPerformanceProfile() {
    if (isLowPowerMode() || (isMobile() && getDeviceMemory() < 2)) {
      return 'low';
    } else if (isMobile() || getDeviceMemory() < 4) {
      return 'medium';
    }
    return 'high';
  }

  // Public API
  return {
    hasWebGL,
    isMobile,
    isLowPowerMode,
    shouldOffer3D,
    getOptimalImageFormat,
    getPerformanceProfile,
    capabilities: {
      webgl: hasWebGL(),
      mobile: isMobile(),
      lowPower: isLowPowerMode(),
      memory: getDeviceMemory(),
      networkSpeed: hasGoodNetworkSpeed()
    }
  };
})();

// Apply performance optimizations based on profile
document.addEventListener('DOMContentLoaded', () => {
  const profile = CythereaDeviceProfile.getPerformanceProfile();
  document.body.classList.add(`perf-${profile}`);
  
  // Log capabilities for debugging
  console.log('Cytherea Device Profile:', CythereaDeviceProfile.capabilities);
});
"""
FastAPI Web Application for MindFractal Lab

Provides web interface for:
- Interactive parameter sliders
- Orbit visualization
- Fractal map generation
- Trait-to-parameter mapping

Run with: uvicorn extensions.webapp.app:app --reload
Or in Termux: python -m uvicorn extensions.webapp.app:app --host 0.0.0.0 --port 8000
"""

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("WARNING: FastAPI not available. Install with: pip install fastapi uvicorn")

import sys
from pathlib import Path
import numpy as np
import io
import base64

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mindfractal.model import FractalDynamicsModel
from mindfractal.simulate import simulate_orbit
from mindfractal.visualize import plot_orbit
from mindfractal.fractal_map import generate_fractal_map, plot_fractal_map
from extensions.psychomapping.trait_to_c import traits_to_parameters

if FASTAPI_AVAILABLE:
    app = FastAPI(title="MindFractal Lab", version="0.1.0")

    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/simulate")
    async def simulate(c1: float = 0.1, c2: float = 0.1):
        """Simulate orbit and return image as base64"""
        c = np.array([c1, c2])
        model = FractalDynamicsModel(c=c)
        x0 = np.array([0.5, 0.5])

        # Generate plot
        fig = plot_orbit(model, x0, n_steps=1000, save_path=None)

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()

        return {"image": f"data:image/png;base64,{img_base64}"}

    @app.get("/fractal")
    async def fractal(resolution: int = 200):
        """Generate fractal map"""
        fractal_data = generate_fractal_map(
            c1_range=(-1.0, 1.0),
            c2_range=(-1.0, 1.0),
            resolution=resolution,
            max_steps=200
        )

        fig = plot_fractal_map(
            fractal_data,
            c1_range=(-1.0, 1.0),
            c2_range=(-1.0, 1.0),
            save_path=None
        )

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()

        return {"image": f"data:image/png;base64,{img_base64}"}

    @app.get("/traits_to_params")
    async def traits(
        openness: float = 0.5,
        volatility: float = 0.5,
        integration: float = 0.5,
        focus: float = 0.5
    ):
        """Map traits to parameters"""
        traits_dict = {
            'openness': openness,
            'volatility': volatility,
            'integration': integration,
            'focus': focus
        }

        c = traits_to_parameters(traits_dict)

        return {
            "c1": float(c[0]),
            "c2": float(c[1]),
            "traits": traits_dict
        }


def main():
    if not FASTAPI_AVAILABLE:
        print("\n=== FastAPI not available ===")
        print("Install with: pip install fastapi uvicorn")
        print("\nFor Termux:")
        print("  pkg install python")
        print("  pip install fastapi uvicorn")
        return

    print("\n=== Starting MindFractal Lab Web Server ===")
    print("Access at: http://localhost:8000")
    print("Or from network: http://0.0.0.0:8000")
    print("\nPress Ctrl+C to stop")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == '__main__':
    main()

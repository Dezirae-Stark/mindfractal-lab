"""
Kivy GUI for MindFractal Lab (Android/Desktop)

Provides sliders for traits, real-time simulation, and visualization.
Falls back to text mode if Kivy unavailable.
"""

try:
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.uix.slider import Slider

    KIVY_AVAILABLE = True
except ImportError:
    KIVY_AVAILABLE = False
    print("WARNING: Kivy not available. Install with: pip install kivy")

import sys
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from extensions.psychomapping.trait_to_c import traits_to_parameters
from mindfractal.model import FractalDynamicsModel
from mindfractal.simulate import simulate_orbit
from mindfractal.visualize import plot_orbit

if KIVY_AVAILABLE:

    class MindFractalApp(App):
        def build(self):
            layout = BoxLayout(orientation="vertical", padding=10, spacing=10)

            # Title
            title = Label(text="MindFractal Lab", size_hint_y=0.1, font_size=24)
            layout.add_widget(title)

            # Trait sliders
            self.sliders = {}
            for trait in ["openness", "volatility", "integration", "focus"]:
                box = BoxLayout(orientation="horizontal", size_hint_y=0.15)
                label = Label(text=trait.capitalize(), size_hint_x=0.3)
                slider = Slider(min=0, max=1, value=0.5, size_hint_x=0.7)
                self.sliders[trait] = slider
                box.add_widget(label)
                box.add_widget(slider)
                layout.add_widget(box)

            # Simulate button
            btn = Button(text="Simulate Orbit", size_hint_y=0.15)
            btn.bind(on_press=self.simulate)
            layout.add_widget(btn)

            # Result label
            self.result_label = Label(text="Adjust traits and press Simulate", size_hint_y=0.2)
            layout.add_widget(self.result_label)

            return layout

        def simulate(self, instance):
            # Get trait values
            traits = {name: slider.value for name, slider in self.sliders.items()}

            # Map to parameters
            c = traits_to_parameters(traits)

            # Simulate
            model = FractalDynamicsModel(c=c)
            x0 = np.array([0.1, 0.1])
            trajectory = simulate_orbit(model, x0, n_steps=1000)

            # Display results
            final_norm = np.linalg.norm(trajectory[-1])
            self.result_label.text = (
                f"c = [{c[0]:.3f}, {c[1]:.3f}]\n"
                f"Final state norm: {final_norm:.3f}\n"
                f"Plot saved to orbit_gui.png"
            )

            # Save plot
            plot_orbit(model, x0, n_steps=1000, save_path="orbit_gui.png")


def run_gui():
    if KIVY_AVAILABLE:
        MindFractalApp().run()
    else:
        print("\n=== MindFractal Lab (Text Mode) ===")
        print("Kivy GUI not available. Using text interface.")
        print("\nInstall Kivy for GUI: pip install kivy")
        print("\nRunning default simulation...")

        from mindfractal.model import FractalDynamicsModel
        from mindfractal.simulate import simulate_orbit

        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])
        trajectory = simulate_orbit(model, x0, n_steps=1000)

        print(f"Simulated orbit with {len(trajectory)} steps")
        print(f"Final state: {trajectory[-1]}")


if __name__ == "__main__":
    run_gui()

# radiation_simple.py
# Minimal script: edit u(theta) only.
# Creates an interactive HTML with a polar plot and a 3D surface,
# and opens it automatically in your default browser.
#
# The plot title is automatically taken from the return statement in u().

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import webbrowser
import inspect
import re

# ---------- EDIT THIS FUNCTION ----------
def u(theta: np.ndarray) -> np.ndarray:
    """Radiation intensity U(theta). Theta in RADIANS, 0..pi."""
    return np.cos(theta)
# ---------------------------------------

OUT_HTML = "radiation_pattern.html"  # output file name


def get_function_expression(func) -> str:
    """Extract the return expression from the function source."""
    src = inspect.getsource(func)
    # Look for 'return ...'
    match = re.search(r"return\s+(.*)", src)
    if match:
        return match.group(1).strip()
    return "U(theta)"


def main():
    expr_str = get_function_expression(u)

    # Theta for polar cut (0..pi/2 == 0..90°)
    theta = np.linspace(0, np.pi/2, 721)
    U = u(theta)

    # Clean up: intensity must be >= 0 and finite
    U = np.asarray(U, dtype=float)
    U[~np.isfinite(U)] = 0.0
    U = np.clip(U, 0.0, None)

    # Normalize for plotting
    umax = float(np.max(U)) if np.max(U) > 0 else 1.0
    U_plot = U / umax

    # Polar data (degrees)
    theta_deg = np.degrees(theta)

    # 3D surface by revolving U(theta) about z-axis
    phi = np.linspace(np.pi/2, np.pi, 721)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    U_on_TH = np.interp(TH[:, 0], theta, U_plot)[:, None] * np.ones_like(PH)
    R = U_on_TH   # radius in linear magnitude

    # Spherical -> Cartesian
    X = R * np.sin(TH) * np.cos(PH)
    Y = R * np.sin(TH) * np.sin(PH)
    Z = R * np.cos(TH)

    # Build figure: left polar, right 3D
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "polar"}, {"type": "surface"}]],
        column_widths=[0.42, 0.58],
        horizontal_spacing=0.08,
        subplot_titles=(f"Polar: U(θ) = {expr_str}", f"3D: U(θ) = {expr_str}")
    )

    # Polar plot (linear)
    fig.add_trace(
        go.Scatterpolar(r=U_plot, theta=theta_deg, mode="lines", name="U(θ)"),
        row=1, col=1
    )
    fig.update_polars(
        radialaxis=dict(range=[0, 1.0], angle=0),
        angularaxis=dict(direction="counterclockwise", rotation=0)
    )

    # 3D plot
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z, surfacecolor=R, showscale=False, name="U"),
        row=1, col=2
    )
    fig.update_scenes(
        xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"
    )

    fig.update_layout(
        title=f"Radiation Pattern: U(θ) = {expr_str} (normalized)",
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False
    )

    # Save and open
    out_path = Path(OUT_HTML)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Wrote: {out_path.resolve()}")

    webbrowser.open(str(out_path.resolve()))


if __name__ == "__main__":
    main()

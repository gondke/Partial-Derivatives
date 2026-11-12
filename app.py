import streamlit as st
import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(page_title="3D Partial & Directional Derivatives", layout="wide")

st.title("Interactive 3D Partial & Directional Derivatives Explorer")
st.markdown(
    "Visualize $z = f(x, y)$, explore partial derivatives along the $x$ and $y$ axes, "
    "and rotate a unit vector $\\mathbf{u}$ to observe the directional derivative $D_{\\mathbf{u}}f(x_0, y_0)$."
)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Function & Grid Settings")
func_str = st.sidebar.text_input(
    "Enter function z = f(x, y) in LaTeX:",
    value=r"\sin(x) \cos(y)",
    help=r"Enter LaTeX (e.g., \sin(x)\cos(y), \frac{x^2 - y^2}{4}, e^{-(x^2+y^2)}) or standard Python math."
)

col_grid1, col_grid2 = st.sidebar.columns(2)
with col_grid1:
    grid_range = st.number_input("Grid Range (±)", min_value=1.0, max_value=20.0, value=3.0, step=0.5)
with col_grid2:
    grid_res = st.slider("Grid Resolution", min_value=20, max_value=100, value=50, step=5)

st.sidebar.header("2. Point Coordinates (x₀, y₀)")
col_pt1, col_pt2 = st.sidebar.columns(2)
with col_pt1:
    x0 = st.number_input("x₀", min_value=-grid_range, max_value=grid_range, value=1.0, step=0.1)
with col_pt2:
    y0 = st.number_input("y₀", min_value=-grid_range, max_value=grid_range, value=0.5, step=0.1)

st.sidebar.header("3. Unit Vector Rotation (u)")
angle_deg = st.sidebar.slider(
    "Direction Angle θ (degrees from +x axis)",
    min_value=0,
    max_value=360,
    value=45,
    step=1
)

st.sidebar.header("4. Visual Overlays")
show_tangent_plane = st.sidebar.checkbox("Show Tangent Plane at (x₀, y₀, z₀)", value=True)
show_partial_dirs = st.sidebar.checkbox("Highlight Partial Derivative Vectors on Plane", value=True)
show_slices = st.sidebar.checkbox("Show Surface Cross-Section Traces", value=True)

# --- SYMBOLIC COMPUTATION (SYMPY WITH LATEX PARSING) ---
x, y = sp.symbols('x y')

def parse_function_input(raw_input: str):
    """Parses LaTeX input first, falling back to standard SymPy syntax if needed."""
    cleaned = raw_input.strip()
    if "=" in cleaned:
        cleaned = cleaned.split("=", 1)[1].strip()
    try:
        expr_parsed = parse_latex(cleaned)
        # Ensure symbol names match our canonical x, y symbols
        subs_map = {}
        for sym in expr_parsed.free_symbols:
            if sym.name == 'x':
                subs_map[sym] = x
            elif sym.name == 'y':
                subs_map[sym] = y
            elif sym.name == 'e':
                subs_map[sym] = sp.E
        return expr_parsed.subs(subs_map)
    except Exception:
        return sp.sympify(cleaned)

try:
    expr = parse_function_input(func_str)

    # Display rendered LaTeX of the parsed function in sidebar for immediate confirmation
    st.sidebar.latex(f"f(x, y) = {sp.latex(expr)}")

    # Compute exact first-order partial derivatives
    fx_expr = sp.diff(expr, x)
    fy_expr = sp.diff(expr, y)

    # Convert symbolic expressions to fast numerical functions
    f_num = sp.lambdify((x, y), expr, modules=['numpy'])
    fx_num = sp.lambdify((x, y), fx_expr, modules=['numpy'])
    fy_num = sp.lambdify((x, y), fy_expr, modules=['numpy'])

    # Evaluate at the selected point (x0, y0)
    z0 = float(f_num(x0, y0))
    fx_val = float(fx_num(x0, y0))
    fy_val = float(fy_num(x0, y0))

except Exception as e:
    st.error(f"Could not parse or evaluate expression at ({x0}, {y0}): {e}")
    st.stop()

# --- DIRECTIONAL DERIVATIVE CALCULATION ---
theta_rad = np.deg2rad(angle_deg)
ux = np.cos(theta_rad)
uy = np.sin(theta_rad)

# Directional derivative: D_u f = fx * ux + fy * uy
dir_deriv = fx_val * ux + fy_val * uy

# --- METRICS & SYMBOLIC DISPLAY ---
st.markdown("### Symbolic & Numerical Analysis")
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(label="Point on Surface (x₀, y₀, z₀)", value=f"({x0:.2f}, {y0:.2f}, {z0:.3f})")
with m2:
    st.metric(label="∂z/∂x at (x₀, y₀)", value=f"{fx_val:.4f}")
    st.latex(f"\\frac{{\\partial z}}{{\\partial x}} = {sp.latex(fx_expr)}")
with m3:
    st.metric(label="∂z/∂y at (x₀, y₀)", value=f"{fy_val:.4f}")
    st.latex(f"\\frac{{\\partial z}}{{\\partial y}} = {sp.latex(fy_expr)}")
with m4:
    st.metric(label=f"Directional Derivative Dᵤf (θ={angle_deg}°)", value=f"{dir_deriv:.4f}")
    st.latex(f"\\mathbf{{u}} = \\langle {ux:.2f}, {uy:.2f} \\rangle")

# --- 3D VISUALIZATION (PLOTLY) ---
lin = np.linspace(-grid_range, grid_range, grid_res)
X, Y = np.meshgrid(lin, lin)

try:
    Z = f_num(X, Y)
    if np.isscalar(Z):
        Z = np.full_like(X, Z)
except Exception as e:
    st.error(f"Error evaluating function over the grid: {e}")
    st.stop()

fig = go.Figure()

# 1. Main Surface z = f(x, y)
fig.add_trace(go.Surface(
    x=X, y=Y, z=Z,
    colorscale='Viridis',
    opacity=0.80,
    name='z = f(x, y)',
    colorbar=dict(title='z value', x=1.02)
))

# 2. Selected Point P(x0, y0, z0)
fig.add_trace(go.Scatter3d(
    x=[x0], y=[y0], z=[z0],
    mode='markers',
    marker=dict(size=7, color='red', symbol='circle'),
    name=f'Point ({x0}, {y0}, {z0:.2f})'
))

# 3. Partial Derivative Cross-Section Traces (x = x0 and y = y0)
if show_slices:
    z_slice_x = f_num(lin, np.full_like(lin, y0))
    if np.isscalar(z_slice_x):
        z_slice_x = np.full_like(lin, z_slice_x)
    fig.add_trace(go.Scatter3d(
        x=lin, y=np.full_like(lin, y0), z=z_slice_x,
        mode='lines',
        line=dict(color='cyan', width=4, dash='dash'),
        name='Surface Trace: y = y₀'
    ))

    z_slice_y = f_num(np.full_like(lin, x0), lin)
    if np.isscalar(z_slice_y):
        z_slice_y = np.full_like(lin, z_slice_y)
    fig.add_trace(go.Scatter3d(
        x=np.full_like(lin, x0), y=lin, z=z_slice_y,
        mode='lines',
        line=dict(color='magenta', width=4, dash='dash'),
        name='Surface Trace: x = x₀'
    ))

vec_scale = grid_range * 0.35

# 4. Tangent Plane at (x0, y0, z0)
if show_tangent_plane:
    px = np.linspace(x0 - vec_scale, x0 + vec_scale, 10)
    py = np.linspace(y0 - vec_scale, y0 + vec_scale, 10)
    PX, PY = np.meshgrid(px, py)
    PZ = z0 + fx_val * (PX - x0) + fy_val * (PY - y0)

    fig.add_trace(go.Surface(
        x=PX, y=PY, z=PZ,
        colorscale=[[0, 'rgba(255, 190, 80, 0.42)'], [1, 'rgba(255, 190, 80, 0.42)']],
        showscale=False,
        name='Tangent Plane'
    ))

# 5. Highlighted First-Order Partial Derivative Tangent Vectors on the Tangent Plane
if show_partial_dirs:
    # Tangent vector in +x direction: <1, 0, fx_val>
    tx_end_x = x0 + vec_scale
    tx_end_y = y0
    tx_end_z = z0 + fx_val * vec_scale

    fig.add_trace(go.Scatter3d(
        x=[x0, tx_end_x], y=[y0, tx_end_y], z=[z0, tx_end_z],
        mode='lines',
        line=dict(color='#0077FF', width=8),
        name='∂z/∂x Tangent Direction ⟨1, 0, fₓ⟩'
    ))
    fig.add_trace(go.Cone(
        x=[tx_end_x], y=[tx_end_y], z=[tx_end_z],
        u=[vec_scale * 0.25], v=[0], w=[fx_val * vec_scale * 0.25],
        colorscale=[[0, '#0077FF'], [1, '#0077FF']],
        showscale=False,
        sizemode='absolute',
        sizeref=grid_range * 0.08,
        anchor='tip',
        name='∂z/∂x Arrow'
    ))

    # Tangent vector in +y direction: <0, 1, fy_val>
    ty_end_x = x0
    ty_end_y = y0 + vec_scale
    ty_end_z = z0 + fy_val * vec_scale

    fig.add_trace(go.Scatter3d(
        x=[x0, ty_end_x], y=[y0, ty_end_y], z=[z0, ty_end_z],
        mode='lines',
        line=dict(color='#D400FF', width=8),
        name='∂z/∂y Tangent Direction ⟨0, 1, fᵧ⟩'
    ))
    fig.add_trace(go.Cone(
        x=[ty_end_x], y=[ty_end_y], z=[ty_end_z],
        u=[0], v=[vec_scale * 0.25], w=[fy_val * vec_scale * 0.25],
        colorscale=[[0, '#D400FF'], [1, '#D400FF']],
        showscale=False,
        sizemode='absolute',
        sizeref=grid_range * 0.08,
        anchor='tip',
        name='∂z/∂y Arrow'
    ))

# 6. Rotating Unit Vector u (horizontal) and Directional Tangent Vector on the Tangent Plane
u_end_x = x0 + ux * vec_scale
u_end_y = y0 + uy * vec_scale
u_end_z = z0

fig.add_trace(go.Scatter3d(
    x=[x0, u_end_x], y=[y0, u_end_y], z=[z0, u_end_z],
    mode='lines+markers',
    line=dict(color='black', width=5),
    marker=dict(size=[3, 5], color='black'),
    name='Unit Vector u (horizontal)'
))

# Directional Tangent Vector on the tangent plane: <ux, uy, D_u f>
t_end_x = x0 + ux * vec_scale
t_end_y = y0 + uy * vec_scale
t_end_z = z0 + dir_deriv * vec_scale

fig.add_trace(go.Scatter3d(
    x=[x0, t_end_x], y=[y0, t_end_y], z=[z0, t_end_z],
    mode='lines',
    line=dict(color='crimson', width=8),
    name='Directional Tangent ⟨uₓ, uᵧ, Dᵤf⟩'
))
fig.add_trace(go.Cone(
    x=[t_end_x], y=[t_end_y], z=[t_end_z],
    u=[ux * vec_scale * 0.25], v=[uy * vec_scale * 0.25], w=[dir_deriv * vec_scale * 0.25],
    colorscale=[[0, 'crimson'], [1, 'crimson']],
    showscale=False,
    sizemode='absolute',
    sizeref=grid_range * 0.08,
    anchor='tip',
    name='Directional Arrow'
))

# Layout & Camera Configuration
# uirevision='constant' locks camera orientation & zoom across slider/coordinate updates
fig.update_layout(
    height=720,
    margin=dict(l=0, r=0, b=0, t=30),
    uirevision='constant',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='cube',
        uirevision='constant'
    ),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.75)')
)

st.plotly_chart(fig, use_container_width=True)

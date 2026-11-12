import streamlit as st
import numpy as np
import sympy as sp
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
    "Enter function z = f(x, y):",
    value="sin(x) * cos(y)",
    help="Use standard Python/SymPy syntax: e.g., x**2 + y**2, exp(-x**2 - y**2), x*y"
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
show_slices = st.sidebar.checkbox("Show Partial Derivative Traces (x & y slices)", value=True)

# --- SYMBOLIC COMPUTATION (SYMPY) ---
x, y = sp.symbols('x y')

try:
    # Parse user input into a SymPy expression
    expr = sp.sympify(func_str)
    
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
    st.error(f"Invalid mathematical expression or domain error at ({x0}, {y0}): {e}")
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
    # Handle constant functions where lambdify returns a scalar
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
    opacity=0.82,
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
    # Trace along x (y held constant at y0)
    z_slice_x = f_num(lin, np.full_like(lin, y0))
    if np.isscalar(z_slice_x): z_slice_x = np.full_like(lin, z_slice_x)
    fig.add_trace(go.Scatter3d(
        x=lin, y=np.full_like(lin, y0), z=z_slice_x,
        mode='lines',
        line=dict(color='cyan', width=5),
        name='Trace: y = y₀ (slope = ∂z/∂x)'
    ))
    
    # Trace along y (x held constant at x0)
    z_slice_y = f_num(np.full_like(lin, x0), lin)
    if np.isscalar(z_slice_y): z_slice_y = np.full_like(lin, z_slice_y)
    fig.add_trace(go.Scatter3d(
        x=np.full_like(lin, x0), y=lin, z=z_slice_y,
        mode='lines',
        line=dict(color='magenta', width=5),
        name='Trace: x = x₀ (slope = ∂z/∂y)'
    ))

# 4. Tangent Plane at (x0, y0, z0)
if show_tangent_plane:
    plane_span = grid_range * 0.35
    px = np.linspace(x0 - plane_span, x0 + plane_span, 10)
    py = np.linspace(y0 - plane_span, y0 + plane_span, 10)
    PX, PY = np.meshgrid(px, py)
    PZ = z0 + fx_val * (PX - x0) + fy_val * (PY - y0)
    
    fig.add_trace(go.Surface(
        x=PX, y=PY, z=PZ,
        colorscale=[[0, 'rgba(255, 165, 0, 0.45)'], [1, 'rgba(255, 165, 0, 0.45)']],
        showscale=False,
        name='Tangent Plane'
    ))

# 5. Unit Vector u (projected in xy-plane at z = z0) and Directional Tangent Vector
vec_scale = grid_range * 0.35

# Horizontal Unit Vector u = <ux, uy, 0> starting at (x0, y0, z0)
u_end_x = x0 + ux * vec_scale
u_end_y = y0 + uy * vec_scale
u_end_z = z0

fig.add_trace(go.Scatter3d(
    x=[x0, u_end_x], y=[y0, u_end_y], z=[z0, u_end_z],
    mode='lines+markers',
    line=dict(color='black', width=6),
    marker=dict(size=[3, 6], color='black'),
    name='Unit Direction Vector u (horizontal)'
))

# Tangent Vector along direction u: <ux, uy, D_u f> scaled
t_end_x = x0 + ux * vec_scale
t_end_y = y0 + uy * vec_scale
t_end_z = z0 + dir_deriv * vec_scale

fig.add_trace(go.Scatter3d(
    x=[x0, t_end_x], y=[y0, t_end_y], z=[z0, t_end_z],
    mode='lines+markers',
    line=dict(color='crimson', width=7),
    marker=dict(size=[3, 7], color='crimson'),
    name='Directional Slope Vector (rise = Dᵤf)'
))

# Layout & Camera Configuration
fig.update_layout(
    height=700,
    margin=dict(l=0, r=0, b=0, t=30),
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='cube'
    ),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.7)')
)

st.plotly_chart(fig, use_container_width=True)

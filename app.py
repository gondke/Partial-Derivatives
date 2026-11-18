import streamlit as st
import numpy as np
from scipy.optimize import root
import sympy as sp
from sympy.parsing.latex import parse_latex
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(page_title="3D Partial & Directional Derivatives", layout="wide")

st.title("Interactive 3D Partial, Directional & Lagrange Multiplier Explorer")
st.markdown(
    "Visualize $z = f(x, y)$, explore partial derivatives along the $x$ and $y$ axes, "
    "inspect critical points (maxima, minima, saddle points), rotate a unit vector $\\mathbf{u}$ "
    "to observe $D_{\\mathbf{u}}f(x_0, y_0)$ and $\\nabla f(x_0, y_0)$, or solve constrained optimization problems."
)

# Initialize session state variables so buttons/dropdowns can update coordinates & angle cleanly
if "angle_deg" not in st.session_state:
    st.session_state.angle_deg = 45
if "x0" not in st.session_state:
    st.session_state.x0 = 1.0
if "y0" not in st.session_state:
    st.session_state.y0 = 0.5

# --- SIDEBAR SECTION 0: PROBLEM MODE ---
st.sidebar.header("0. Optimization Mode")
mode = st.sidebar.radio(
    "Select Analysis Mode:",
    options=[
        "Unconstrained (Partial & Directional Derivatives)",
        "Constrained (Method of Lagrange Multipliers)"
    ],
    index=0
)
is_constrained = (mode == "Constrained (Method of Lagrange Multipliers)")

# --- SIDEBAR SECTION 1: FUNCTION, CONSTRAINT & GRID ---
st.sidebar.header("1. Function & Grid Settings")
func_str = st.sidebar.text_input(
    "Enter function z = f(x, y) in LaTeX:",
    value=r"\sin(x) \cos(y)" if not is_constrained else r"x^2 + 2y^2",
    help=r"Enter LaTeX (e.g., \sin(x)\cos(y), x^2 + 2y^2, e^{-(x^2+y^2)}) or standard Python math."
)

constraint_str = ""
if is_constrained:
    constraint_str = st.sidebar.text_input(
        "Enter constraint g(x, y) = 0 (or lhs = rhs):",
        value=r"x^2 + y^2 = 1",
        help=r"Enter an equation like 'x^2 + y^2 = 1' or expression 'x^2 + y^2 - 1' (interpreted as = 0)."
    )

col_grid1, col_grid2 = st.sidebar.columns(2)
with col_grid1:
    grid_range = st.number_input("Grid Range (±)", min_value=1.0, max_value=20.0, value=3.0, step=0.5)
with col_grid2:
    grid_res = st.slider("Grid Resolution", min_value=20, max_value=100, value=50, step=5)

# Keep session coordinates clamped within grid_range if user shrinks the grid
st.session_state.x0 = float(np.clip(st.session_state.x0, -grid_range, grid_range))
st.session_state.y0 = float(np.clip(st.session_state.y0, -grid_range, grid_range))

# --- SYMBOLIC COMPUTATION (SYMPY WITH LATEX PARSING) ---
x, y, lam = sp.symbols('x y lambda')

def _parse_single_latex(text_part: str):
    cleaned = text_part.strip()
    try:
        expr_parsed = parse_latex(cleaned)
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

def parse_function_input(raw_input: str):
    """Parses LaTeX input first, falling

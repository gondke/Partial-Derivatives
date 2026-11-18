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
    """Parses LaTeX input first, falling back to standard SymPy syntax if needed."""
    cleaned = raw_input.strip()
    if "=" in cleaned:
        cleaned = cleaned.split("=", 1)[1].strip()
    return _parse_single_latex(cleaned)

def parse_constraint_input(raw_input: str):
    """Parses constraint g(x, y) = 0 or lhs = rhs into g(x, y) = lhs - rhs."""
    cleaned = raw_input.strip()
    if "=" in cleaned:
        lhs_str, rhs_str = cleaned.split("=", 1)
        return sp.simplify(_parse_single_latex(lhs_str) - _parse_single_latex(rhs_str))
    return _parse_single_latex(cleaned)

try:
    expr = parse_function_input(func_str)
    st.sidebar.latex(f"f(x, y) = {sp.latex(expr)}")

    # Compute exact first- and second-order partial derivatives of f
    fx_expr = sp.diff(expr, x)
    fy_expr = sp.diff(expr, y)
    fxx_expr = sp.diff(fx_expr, x)
    fyy_expr = sp.diff(fy_expr, y)
    fxy_expr = sp.diff(fx_expr, y)

    # Convert symbolic expressions to fast numerical functions
    f_num = sp.lambdify((x, y), expr, modules=['numpy'])
    fx_num = sp.lambdify((x, y), fx_expr, modules=['numpy'])
    fy_num = sp.lambdify((x, y), fy_expr, modules=['numpy'])
    fxx_num = sp.lambdify((x, y), fxx_expr, modules=['numpy'])
    fyy_num = sp.lambdify((x, y), fyy_expr, modules=['numpy'])
    fxy_num = sp.lambdify((x, y), fxy_expr, modules=['numpy'])

    if is_constrained:
        g_expr = parse_constraint_input(constraint_str)
        st.sidebar.latex(f"g(x, y) = {sp.latex(g_expr)} = 0")

        gx_expr = sp.diff(g_expr, x)
        gy_expr = sp.diff(g_expr, y)
        L_expr = expr - lam * g_expr

        # Bordered Hessian matrix for n=2 variables, m=1 constraint:
        # [ 0   gx   gy  ]
        # [ gx  Lxx  Lxy ]
        # [ gy  Lyx  Lyy ]
        Lxx_expr = sp.diff(L_expr, x, x)
        Lyy_expr = sp.diff(L_expr, y, y)
        Lxy_expr = sp.diff(L_expr, x, y)

        bordered_hess_sym = sp.Matrix([
            [0, gx_expr, gy_expr],
            [gx_expr, Lxx_expr, Lxy_expr],
            [gy_expr, Lxy_expr, Lyy_expr]
        ])

        g_num = sp.lambdify((x, y), g_expr, modules=['numpy'])
        gx_num = sp.lambdify((x, y), gx_expr, modules=['numpy'])
        gy_num = sp.lambdify((x, y), gy_expr, modules=['numpy'])
        grad_L_num = sp.lambdify((x, y, lam), [sp.diff(L_expr, x), sp.diff(L_expr, y), -g_expr], modules=['numpy'])
        bordered_hess_num = sp.lambdify((x, y, lam), bordered_hess_sym, modules=['numpy'])

except Exception as e:
    st.error(f"Could not parse expression: {e}")
    st.stop()

# --- CRITICAL POINT FINDERS & CLASSIFIERS ---
def find_unconstrained_critical_points(bound: float):
    """Finds unconstrained critical points in [-bound, bound]^2 via symbolic + numerical solvers
    and classifies them using the Second Derivative Discriminant Test."""
    candidates = []

    try:
        sols = sp.solve([fx_expr, fy_expr], (x, y), dict=True)
        for s in sols:
            if x in s and y in s:
                xv = complex(s[x].evalf())
                yv = complex(s[y].evalf())
                if abs(xv.imag) < 1e-7 and abs(yv.imag) < 1e-7:
                    candidates.append((float(xv.real), float(yv.real)))
    except Exception:
        pass

    def grad_vec(pt):
        try:
            gx = float(fx_num(pt[0], pt[1]))
            gy = float(fy_num(pt[0], pt[1]))
            if not (np.isfinite(gx) and np.isfinite(gy)):
                return np.array([1e6, 1e6])
            return np.array([gx, gy], dtype=float)
        except Exception:
            return np.array([1e6, 1e6])

    def hess_mat(pt):
        try:
            hxx = float(fxx_num(pt[0], pt[1]))
            hyy = float(fyy_num(pt[0], pt[1]))
            hxy = float(fxy_num(pt[0], pt[1]))
            return np.array([[hxx, hxy], [hxy, hyy]], dtype=float)
        except Exception:
            return np.eye(2)

    seeds = np.linspace(-bound, bound, 9)
    for sx in seeds:
        for sy in seeds:
            res = root(grad_vec, x0=np.array([sx, sy]), jac=hess_mat, method='hybr', tol=1e-10)
            if res.success:
                rx, ry = float(res.x[0]), float(res.x[1])
                if np.linalg.norm(grad_vec([rx, ry])) < 1e-6:
                    candidates.append((rx, ry))

    unique_pts = []
    for cx, cy in candidates:
        if -bound - 1e-5 <= cx <= bound + 1e-5 and -bound - 1e-5 <= cy <= bound + 1e-5:
            cx = 0.0 if abs(cx) < 1e-7 else cx
            cy = 0.0 if abs(cy) < 1e-7 else cy
            if not any(np.hypot(cx - ux, cy - uy) < 1e-3 for ux, uy in unique_pts):
                unique_pts.append((cx, cy))

    unique_pts.sort(key=lambda p: (round(p[0], 4), round(p[1], 4)))

    classified = []
    for cx, cy in unique_pts:
        try:
            cz = float(f_num(cx, cy))
            hxx = float(fxx_num(cx, cy))
            hyy = float(fyy_num(cx, cy))
            hxy = float(fxy_num(cx, cy))
            if not all(np.isfinite([cz, hxx, hyy, hxy])):
                continue
            disc = hxx * hyy - hxy ** 2
            if disc > 1e-6:
                cp_type = "Local Maximum" if hxx < 0 else "Local Minimum"
            elif disc < -1e-6:
                cp_type = "Saddle Point"
            else:
                cp_type = "Inconclusive (D ≈ 0)"

            classified.append({
                "x": cx,
                "y": cy,
                "z": cz,
                "type": cp_type,
                "D": disc,
                "fxx": hxx,
                "fyy": hyy,
                "fxy": hxy
            })
        except Exception:
            continue

    return classified

def find_constrained_critical_points(bound: float):
    """Finds constrained critical points of f(x, y) subject to g(x, y) = 0 via Lagrange Multipliers
    and classifies them using the Bordered Hessian determinant |H̄₃|."""
    raw_solutions = []

    # 1. Exact Symbolic Solve of grad(L) = 0
    try:
        eqs = [sp.diff(L_expr, x), sp.diff(L_expr, y), -g_expr]
        sols = sp.solve(eqs, (x, y, lam), dict=True)
        for s in sols:
            if x in s and y in s and lam in s:
                xv = complex(s[x].evalf())
                yv = complex(s[y].evalf())
                lv = complex(s[lam].evalf())
                if abs(xv.imag) < 1e-7 and abs(yv.imag) < 1e-7 and abs(lv.imag) < 1e-7:
                    raw_solutions.append((float(xv.real), float(yv.real), float(lv.real)))
    except Exception:
        pass

    # 2. Multi-start Numerical Root Finding on [Lx, Ly, -g] = [0, 0, 0]
    def residual_fn(vec):
        try:
            vals = np.array(grad_L_num(vec[0], vec[1], vec[2]), dtype=float)
            if not np.all(np.isfinite(vals)):
                return np.array([1e6, 1e6, 1e6])
            return vals
        except Exception:
            return np.array([1e6, 1e6, 1e6])

    seeds = np.linspace(-bound, bound, 8)
    for sx in seeds:
        for sy in seeds:
            for slam in (-1.0, 1.0):
                res = root(residual_fn, x0=np.array([sx, sy, slam]), method='hybr', tol=1e-10)
                if res.success and np.linalg.norm(residual_fn(res.x)) < 1e-6:
                    raw_solutions.append((float(res.x[0]), float(res.x[1]), float(res.x[2])))

    unique_sols = []
    for cx, cy, cl in raw_solutions:
        if -bound - 1e-4 <= cx <= bound + 1e-4 and -bound - 1e-4 <= cy <= bound + 1e-4:
            cx = 0.0 if abs(cx) < 1e-7 else cx
            cy = 0.0 if abs(cy) < 1e-7 else cy
            cl = 0.0 if abs(cl) < 1e-7 else cl
            if not any(np.hypot(cx - ux, cy - uy) < 1e-3 for ux, uy, _ in unique_sols):
                unique_sols.append((cx, cy, cl))

    unique_sols.sort(key=lambda p: (round(p[0], 4), round(p[1], 4)))

    classified = []
    for cx, cy, cl in unique_sols:
        try:
            cz = float(f_num(cx, cy))
            H_bar = np.array(bordered_hess_num(cx, cy, cl), dtype=float)
            if not (np.isfinite(cz) and np.all(np.isfinite(H_bar))):
                continue
            det_H3 = float(np.linalg.det(H_bar))
            # For n=2 variables and m=1 constraint:
            # |H̄₃| > 0 => Constrained Local Maximum
            # |H̄₃| < 0 => Constrained Local Minimum
            if det_H3 > 1e-6:
                cp_type = "Constrained Maximum"
            elif det_H3 < -1e-6:
                cp_type = "Constrained Minimum"
            else:
                cp_type = "Inconclusive (|H̄₃| ≈ 0)"

            classified.append({
                "x": cx,
                "y": cy,
                "z": cz,
                "lambda": cl,
                "type": cp_type,
                "det_H3": det_H3,
                "H_bar": H_bar
            })
        except Exception:
            continue

    return classified

if is_constrained:
    critical_points = find_constrained_critical_points(grid_range)
else:
    critical_points = find_unconstrained_critical_points(grid_range)

# --- SIDEBAR SECTION 2: CRITICAL POINTS & ACTIVE POINT (x0, y0) ---
st.sidebar.header("2. Critical Points & Point (x₀, y₀)")

if critical_points:
    if is_constrained:
        cp_options = [
            f"#{i+1}: {cp['type']} at ({cp['x']:.3f}, {cp['y']:.3f}), z = {cp['z']:.3f}, λ = {cp['lambda']:.3f}"
            for i, cp in enumerate(critical_points)
        ]
    else:
        cp_options = [
            f"#{i+1}: {cp['type']} at ({cp['x']:.3f}, {cp['y']:.3f}), z = {cp['z']:.3f}"
            for i, cp in enumerate(critical_points)
        ]

    selected_cp_idx = st.sidebar.selectbox(
        f"Detected Critical Points ({len(critical_points)} in grid):",
        options=range(len(critical_points)),
        format_func=lambda idx: cp_options[idx]
    )

    def jump_to_selected_cp():
        chosen = critical_points[selected_cp_idx]
        st.session_state.x0 = float(np.clip(chosen["x"], -grid_range, grid_range))
        st.session_state.y0 = float(np.clip(chosen["y"], -grid_range, grid_range))

    st.sidebar.button(
        "Jump (x₀, y₀) to Selected Critical Point",
        on_click=jump_to_selected_cp,
        help="Moves the active point (x₀, y₀) onto the selected critical point."
    )
else:
    st.sidebar.info("No critical points detected inside the current grid range.")

col_pt1, col_pt2 = st.sidebar.columns(2)
with col_pt1:
    x0 = st.number_input("x₀", min_value=-grid_range, max_value=grid_range, key="x0", step=0.1)
with col_pt2:
    y0 = st.number_input("y₀", min_value=-grid_range, max_value=grid_range, key="y0", step=0.1)

# Evaluate function and derivatives at active point (x0, y0)
try:
    z0 = float(f_num(x0, y0))
    fx_val = float(fx_num(x0, y0))
    fy_val = float(fy_num(x0, y0))
except Exception as e:
    st.error(f"Error evaluating function at ({x0}, {y0}): {e}")
    st.stop()

# Compute gradient magnitude and steepest ascent angle before rendering slider
grad_mag = float(np.hypot(fx_val, fy_val))
steepest_angle_deg = int(np.round(np.rad2deg(np.arctan2(fy_val, fx_val))) % 360)

def snap_to_steepest_ascent():
    st.session_state.angle_deg = steepest_angle_deg

# --- SIDEBAR SECTION 3 & 4 ---
st.sidebar.header("3. Unit Vector Rotation (u)")
st.sidebar.button(
    f"Snap θ to Steepest Ascent ({steepest_angle_deg}°)",
    on_click=snap_to_steepest_ascent,
    disabled=(grad_mag < 1e-9),
    help="Aligns the unit vector u with the gradient vector ∇f(x₀, y₀) to maximize Dᵤf."
)

angle_deg = st.sidebar.slider(
    "Direction Angle θ (degrees from +x axis)",
    min_value=0,
    max_value=360,
    key="angle_deg",
    step=1
)

st.sidebar.header("4. Visual Overlays")
show_critical_pts = st.sidebar.checkbox("Mark All Critical Points on Surface", value=True)
if is_constrained:
    show_constraint_curve = st.sidebar.checkbox("Highlight Function Over Constraint g(x, y) = 0", value=True)
show_tangent_plane = st.sidebar.checkbox("Show Tangent Plane at (x₀, y₀, z₀)", value=True)
show_partial_dirs = st.sidebar.checkbox("Highlight Partial Derivative Vectors on Plane", value=True)
show_gradient_xy = st.sidebar.checkbox("Show Gradient Vector ∇f on xy-Plane", value=True)
show_slices = st.sidebar.checkbox("Show Surface Cross-Section Traces", value=not is_constrained)

# --- DIRECTIONAL DERIVATIVE CALCULATION ---
theta_rad = np.deg2rad(angle_deg)
ux = np.cos(theta_rad)
uy = np.sin(theta_rad)
dir_deriv = fx_val * ux + fy_val * uy

# --- METRICS & SYMBOLIC DISPLAY ---
st.markdown("### Symbolic & Numerical Analysis")
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric(label="Point on Surface (x₀, y₀, z₀)", value=f"({x0:.2f}, {y0:.2f}, {z0:.3f})")
with m2:
    st.metric(label="∂z/∂x at (x₀, y₀)", value=f"{fx_val:.4f}")
    st.latex(f"\\frac{{\\partial z}}{{\\partial x}} = {sp.latex(fx_expr)}")
with m3:
    st.metric(label="∂z/∂y at (x₀, y₀)", value=f"{fy_val:.4f}")
    st.latex(f"\\frac{{\\partial z}}{{\\partial y}} = {sp.latex(fy_expr)}")
with m4:
    st.metric(label="Gradient Magnitude ‖∇f‖", value=f"{grad_mag:.4f}")
    st.latex(f"\\nabla f = \\langle {fx_val:.2f}, {fy_val:.2f} \\rangle")
with m5:
    st.metric(label=f"Directional Deriv. Dᵤf (θ={angle_deg}°)", value=f"{dir_deriv:.4f}")
    st.latex(f"\\mathbf{{u}} = \\langle {ux:.2f}, {uy:.2f} \\rangle")

# --- LAGRANGE MULTIPLIER & BORDERED HESSIAN PANEL (WHEN CONSTRAINED MODE IS ACTIVE) ---
if is_constrained:
    with st.expander("Lagrange Multiplier Formulation & Bordered Hessian Matrix Test", expanded=True):
        l_col1, l_col2 = st.columns([1.1, 1])
        with l_col1:
            st.markdown("**Lagrangian Function & Stationarity System ($\\nabla \\mathcal{L} = \\mathbf{0}$):**")
            st.latex(f"\\mathcal{{L}}(x, y, \\lambda) = {sp.latex(L_expr)}")
            st.latex(
                r"\begin{cases} "
                f"{sp.latex(sp.diff(L_expr, x))} = 0 \\\\ "
                f"{sp.latex(sp.diff(L_expr, y))} = 0 \\\\ "
                f"{sp.latex(g_expr)} = 0 "
                r"\end{cases}"
            )
            st.markdown("**Symbolic Bordered Hessian Matrix $\\bar{H}$:**")
            st.latex(r"\bar{H} = " + sp.latex(bordered_hess_sym))

        with l_col2:
            if critical_points:
                st.markdown("**Constrained Critical Points & Bordered Hessian Classification:**")
                c_rows = [
                    {
                        "#": i + 1,
                        "Classification": cp["type"],
                        "x": round(cp["x"], 4),
                        "y": round(cp["y"], 4),
                        "z = f(x, y)": round(cp["z"], 4),
                        "λ": round(cp["lambda"], 4),
                        "|H̄₃|": round(cp["det_H3"], 4),
                    }
                    for i, cp in enumerate(critical_points)
                ]
                st.dataframe(c_rows, use_container_width=True, hide_index=True)

                chosen_cp = critical_points[selected_cp_idx]
                st.markdown(f"**Evaluated $\\bar{{H}}$ at Critical Point #{selected_cp_idx + 1}:**")
                st.latex(
                    r"\bar{H}(P^*) = "
                    + sp.latex(sp.Matrix(np.round(chosen_cp["H_bar"], 4)))
                    + f" \\implies |\\bar{{H}}_3| = {chosen_cp['det_H3']:.4f}"
                )
                if chosen_cp["type"] == "Constrained Maximum":
                    st.success(f"$\vert{}\\bar{{H}}_3\vert{} > 0 \\implies$ **{chosen_cp['type']}** ($z = {chosen_cp['z']:.4f}$)")
                elif chosen_cp["type"] == "Constrained Minimum":
                    st.success(f"$\vert{}\\bar{{H}}_3\vert{} < 0 \\implies$ **{chosen_cp['type']}** ($z = {chosen_cp['z']:.4f}$)")
                else:
                    st.warning(f"$\vert{}\\bar{{H}}_3\vert{} \\approx 0 \\implies$ **{chosen_cp['type']}**")
            else:
                st.info("No constrained critical points found inside the current grid range.")
else:
    # Unconstrained Critical Points Summary Expander
    with st.expander(f"Critical Points Classification Table ({len(critical_points)} detected in grid)", expanded=False):
        if critical_points:
            table_rows = [
                {
                    "Point #": i + 1,
                    "Classification": cp["type"],
                    "x": round(cp["x"], 4),
                    "y": round(cp["y"], 4),
                    "z = f(x, y)": round(cp["z"], 4),
                    "Discriminant D = fxx·fyy - fxy²": round(cp["D"], 4),
                    "fxx": round(cp["fxx"], 4),
                    "fyy": round(cp["fyy"], 4),
                    "fxy": round(cp["fxy"], 4),
                }
                for i, cp in enumerate(critical_points)
            ]
            st.dataframe(table_rows, use_container_width=True, hide_index=True)
        else:
            st.write("No critical points found inside the current grid bounds.")

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
    opacity=0.72 if is_constrained else 0.80,
    name='z = f(x, y)',
    colorbar=dict(title='z value', x=1.02)
))

# 1b. If Constrained Mode: Extract g(x, y) = 0 and highlight f(x, y) directly over the constraint
if is_constrained and show_constraint_curve:
    def extract_contour_segments(Xg, Yg, Gg):
        """Extracts 0-contour segments of Gg = 0 using marching squares."""
        segs_x, segs_y = [], []
        rows, cols = Gg.shape
        for r in range(rows - 1):
            for c in range(cols - 1):
                vals = [Gg[r, c], Gg[r, c+1], Gg[r+1, c+1], Gg[r+1, c]]
                if not all(np.isfinite(vals)):
                    continue
                pts = []
                corners = [
                    (Xg[r, c], Yg[r, c], vals[0]),
                    (Xg[r, c+1], Yg[r, c+1], vals[1]),
                    (Xg[r+1, c+1], Yg[r+1, c+1], vals[2]),
                    (Xg[r+1, c], Yg[r+1, c], vals[3]),
                ]
                for i1, i2 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                    x1, y1, v1 = corners[i1]
                    x2, y2, v2 = corners[i2]
                    if (v1 <= 0 and v2 > 0) or (v2 <= 0 and v1 > 0):
                        t = abs(v1) / (abs(v1) + abs(v2))
                        pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
                if len(pts) >= 2:
                    segs_x.extend([pts[0][0], pts[1][0], None])
                    segs_y.extend([pts[0][1], pts[1][1], None])
        return np.array(segs_x, dtype=object), np.array(segs_y, dtype=object)

    fine_lin = np.linspace(-grid_range, grid_range, 140)
    Xf, Yf = np.meshgrid(fine_lin, fine_lin)
    Gf = g_num(Xf, Yf)
    if not np.isscalar(Gf):
        cx, cy = extract_contour_segments(Xf, Yf, Gf)
        if len(cx) > 0:
            cz = [None if px_v is None else float(f_num(px_v, py_v)) for px_v, py_v in zip(cx, cy)]

            # Highlighted curve of f(x, y) directly over constraint g(x, y) = 0
            fig.add_trace(go.Scatter3d(
                x=cx, y=cy, z=cz,
                mode='lines',
                line=dict(color='#FF1744', width=10),
                name='Function over Constraint: f(x, y) on g(x, y) = 0'
            ))

            # Constraint curve g(x, y) = 0 projected onto base plane (z = min(Z))
            z_base = float(np.nanmin(Z))
            cz_base = [None if px_v is None else z_base for px_v in cx]
            fig.add_trace(go.Scatter3d(
                x=cx, y=cy, z=cz_base,
                mode='lines',
                line=dict(color='#00E5FF', width=5, dash='dash'),
                name='Constraint Curve g(x, y) = 0 (projected)'
            ))

# 2. Critical Points Markers (grouped by classification)
if show_critical_pts and critical_points:
    cp_styles = {
        "Local Maximum": dict(color="#FFD700", symbol="diamond", size=8),
        "Local Minimum": dict(color="#00E5FF", symbol="diamond", size=8),
        "Saddle Point": dict(color="#FF4500", symbol="cross", size=8),
        "Inconclusive (D ≈ 0)": dict(color="#CCCCCC", symbol="square", size=7),
        "Constrained Maximum": dict(color="#FFD700", symbol="diamond", size=10),
        "Constrained Minimum": dict(color="#00FF7F", symbol="diamond", size=10),
        "Inconclusive (|H̄₃| ≈ 0)": dict(color="#CCCCCC", symbol="square", size=8),
    }
    for cp_type, style in cp_styles.items():
        pts = [cp for cp in critical_points if cp["type"] == cp_type]
        if pts:
            fig.add_trace(go.Scatter3d(
                x=[p["x"] for p in pts],
                y=[p["y"] for p in pts],
                z=[p["z"] for p in pts],
                mode='markers+text',
                text=[f"{cp_type}<br>({p['x']:.2f}, {p['y']:.2f}, {p['z']:.2f})" for p in pts],
                textposition="top center",
                textfont=dict(size=10, color="black"),
                marker=dict(
                    size=style["size"],
                    color=style["color"],
                    symbol=style["symbol"],
                    line=dict(width=1.5, color="black")
                ),
                name=f"{cp_type} ({len(pts)})"
            ))

# 3. Selected Active Point P(x0, y0, z0)
fig.add_trace(go.Scatter3d(
    x=[x0], y=[y0], z=[z0],
    mode='markers',
    marker=dict(size=7, color='red', symbol='circle'),
    name=f'Active Point ({x0:.2f}, {y0:.2f}, {z0:.2f})'
))

# 4. Partial Derivative Cross-Section Traces (x = x0 and y = y0)
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

# 5. Tangent Plane at (x0, y0, z0)
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

# 6. Highlighted First-Order Partial Derivative Tangent Vectors on the Tangent Plane
if show_partial_dirs:
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

# 7. 3D Gradient Vector Arrow on the xy-plane (z = 0)
if show_gradient_xy:
    fig.add_trace(go.Scatter3d(
        x=[x0, x0], y=[y0, y0], z=[0, z0],
        mode='lines+markers',
        line=dict(color='#00A86B', width=3, dash='dot'),
        marker=dict(size=[5, 0], color='#00A86B'),
        name='Projection to xy-plane (x₀, y₀, 0)'
    ))

    if grad_mag > 1e-9:
        grad_visual_scale = vec_scale if grad_mag <= 2.5 else (vec_scale * 2.5 / grad_mag)
        gx_end = x0 + fx_val * grad_visual_scale
        gy_end = y0 + fy_val * grad_visual_scale

        fig.add_trace(go.Scatter3d(
            x=[x0, gx_end], y=[y0, gy_end], z=[0, 0],
            mode='lines',
            line=dict(color='#00A86B', width=8),
            name='Gradient Vector ∇f = ⟨fₓ, fᵧ, 0⟩ (xy-plane)'
        ))
        fig.add_trace(go.Cone(
            x=[gx_end], y=[gy_end], z=[0],
            u=[fx_val * grad_visual_scale * 0.25],
            v=[fy_val * grad_visual_scale * 0.25],
            w=[0],
            colorscale=[[0, '#00A86B'], [1, '#00A86B']],
            showscale=False,
            sizemode='absolute',
            sizeref=grid_range * 0.08,
            anchor='tip',
            name='Gradient Arrow'
        ))

# 8. Rotating Unit Vector u (horizontal at z = z0) and Directional Tangent Vector
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

t_end_x = x0 + ux * vec_scale
t_end_y = y0 + uy * vec_scale
t

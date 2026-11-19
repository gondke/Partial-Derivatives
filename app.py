import streamlit as st
import numpy as np
from scipy.optimize import root
import sympy as sp
from sympy.parsing.latex import parse_latex
import plotly.graph_objects as go

# Page Configuration (must be the first Streamlit command)
st.set_page_config(page_title="3D Calculus & Geometric System Solver", layout="wide")


# ==============================================================================
# PAGE 1: EXISTING 3D PARTIAL, DIRECTIONAL & LAGRANGE MULTIPLIER EXPLORER
# (Kept 100% unchanged from original setup)
# ==============================================================================
def page_derivatives_and_lagrange():
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

    selected_cp_idx = 0
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
                        st.success(f"$|\\bar{{H}}_3| > 0 \\implies$ **{chosen_cp['type']}** ($z = {chosen_cp['z']:.4f}$)")
                    elif chosen_cp["type"] == "Constrained Minimum":
                        st.success(f"$|\\bar{{H}}_3| < 0 \\implies$ **{chosen_cp['type']}** ($z = {chosen_cp['z']:.4f}$)")
                    else:
                        st.warning(f"$|\\bar{{H}}_3| \\approx 0 \\implies$ **{chosen_cp['type']}**")
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

    # Layout & Camera Configuration (uirevision locks camera orientation across updates)
    fig.update_layout(
        height=740,
        margin=dict(l=0, r=0, b=0, t=30),
        uirevision='constant',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
            uirevision='constant'
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.78)')
    )

    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PAGE 2: GEOMETRIC SYSTEM OF EQUATIONS SOLVER (2 & 3 VARIABLES)
# ==============================================================================
def page_geometric_system_solver():
    st.title("Geometric System of Equations Solver (2D & 3D)")
    st.markdown(
        "Visualize systems of **2 variables** $(x, y)$ as intersecting curves/lines in the plane, "
        "or systems of **3 variables** $(x, y, z)$ as intersecting surfaces/planes in 3D space."
    )

    x, y, z = sp.symbols("x y z")

    def parse_eq_to_expr(raw_eq: str, allowed_vars):
        """Converts 'lhs = rhs' or 'expr' into symbolic expression lhs - rhs = 0."""
        cleaned = raw_eq.strip()
        if not cleaned:
            raise ValueError("Equation cannot be empty.")

        def _parse_side(side_str: str):
            try:
                parsed = parse_latex(side_str.strip())
                subs_map = {}
                for sym in parsed.free_symbols:
                    if sym.name == "x":
                        subs_map[sym] = x
                    elif sym.name == "y":
                        subs_map[sym] = y
                    elif sym.name == "z":
                        subs_map[sym] = z
                    elif sym.name == "e":
                        subs_map[sym] = sp.E
                return parsed.subs(subs_map)
            except Exception:
                return sp.sympify(side_str.strip())

        if "=" in cleaned:
            lhs_s, rhs_s = cleaned.split("=", 1)
            expr = sp.simplify(_parse_side(lhs_s) - _parse_side(rhs_s))
        else:
            expr = sp.simplify(_parse_side(cleaned))

        extra_syms = expr.free_symbols - set(allowed_vars)
        if extra_syms:
            raise ValueError(f"Unexpected symbol(s) {extra_syms}. Allowed: {allowed_vars}")
        return expr

    # --- SIDEBAR CONTROLS FOR SYSTEM SOLVER ---
    st.sidebar.header("1. System Dimension")
    sys_dim = st.sidebar.radio(
        "Number of Variables:",
        options=["2 Variables (x, y) — 2D Plane", "3 Variables (x, y, z) — 3D Space"],
        index=0,
        key="sys_dim_radio"
    )
    is_2d = sys_dim.startswith("2 Variables")

    st.sidebar.header("2. Presets & Equations")
    if is_2d:
        preset_2d = st.sidebar.selectbox(
            "Load Example System (2D):",
            options=[
                "Custom",
                "Linear: Unique Intersection",
                "Linear: Parallel Lines (No Solution)",
                "Linear: Coincident Lines (Infinite Solutions)",
                "Nonlinear: Circle & Line",
                "Nonlinear: Parabola & Circle"
            ],
            index=1,
            key="preset_2d"
        )
        defaults_2d = {
            "Custom": (r"2x + y = 4", r"x - y = 2"),
            "Linear: Unique Intersection": (r"2x + y = 4", r"x - y = 2"),
            "Linear: Parallel Lines (No Solution)": (r"x + 2y = 4", r"2x + 4y = -4"),
            "Linear: Coincident Lines (Infinite Solutions)": (r"x - y = 1", r"2x - 2y = 2"),
            "Nonlinear: Circle & Line": (r"x^2 + y^2 = 9", r"y = x + 1"),
            "Nonlinear: Parabola & Circle": (r"y = x^2 - 2", r"x^2 + y^2 = 4"),
        }
        d_eq1, d_eq2 = defaults_2d[preset_2d]
        eq1_str = st.sidebar.text_input("Equation 1:", value=d_eq1, key=f"eq2d_1_{preset_2d}")
        eq2_str = st.sidebar.text_input("Equation 2:", value=d_eq2, key=f"eq2d_2_{preset_2d}")

        col_r1, col_r2 = st.sidebar.columns(2)
        with col_r1:
            bound = st.number_input("Axis Range (±)", min_value=2.0, max_value=50.0, value=6.0, step=1.0, key="bound_2d")
        with col_r2:
            res_2d = st.slider("Grid Resolution", min_value=80, max_value=300, value=180, step=20, key="res_2d")

        # Parse & Solve 2D System
        try:
            f1 = parse_eq_to_expr(eq1_str, (x, y))
            f2 = parse_eq_to_expr(eq2_str, (x, y))
        except Exception as e:
            st.error(f"Equation parsing error: {e}")
            st.stop()

        st.sidebar.latex(f"\\text{{Eq 1: }} {sp.latex(f1)} = 0")
        st.sidebar.latex(f"\\text{{Eq 2: }} {sp.latex(f2)} = 0")

        f1_num = sp.lambdify((x, y), f1, modules=["numpy"])
        f2_num = sp.lambdify((x, y), f2, modules=["numpy"])

        # Check if system is linear in (x, y)
        is_linear_2d = f1.is_polynomial(x, y) and f2.is_polynomial(x, y) and (
            sp.Poly(f1, x, y).total_degree() <= 1 and sp.Poly(f2, x, y).total_degree() <= 1
        )

        # Solve symbolically + numerically
        solutions_2d = []
        symbolic_status = ""

        try:
            sym_sols = sp.solve([f1, f2], (x, y), dict=True)
            for s in sym_sols:
                if x in s and y in s:
                    if not s[x].free_symbols and not s[y].free_symbols:
                        xv = complex(s[x].evalf())
                        yv = complex(s[y].evalf())
                        if abs(xv.imag) < 1e-7 and abs(yv.imag) < 1e-7:
                            solutions_2d.append((float(xv.real), float(yv.real)))
                    else:
                        symbolic_status = f"Infinitely many solutions (parametric: $x = {sp.latex(s[x])},\\; y = {sp.latex(s[y])}$)"
        except Exception:
            pass

        # Multi-start numerical root finder to catch any nonlinear intersections missed symbolically
        def res_2d_fn(pt):
            try:
                v1 = float(f1_num(pt[0], pt[1]))
                v2 = float(f2_num(pt[0], pt[1]))
                if not (np.isfinite(v1) and np.isfinite(v2)):
                    return np.array([1e6, 1e6])
                return np.array([v1, v2], dtype=float)
            except Exception:
                return np.array([1e6, 1e6])

        seeds = np.linspace(-bound, bound, 10)
        for sx in seeds:
            for sy in seeds:
                r = root(res_2d_fn, x0=np.array([sx, sy]), method="hybr", tol=1e-10)
                if r.success and np.linalg.norm(res_2d_fn(r.x)) < 1e-6:
                    solutions_2d.append((float(r.x[0]), float(r.x[1])))

        unique_sols_2d = []
        for sx, sy in solutions_2d:
            sx = 0.0 if abs(sx) < 1e-7 else sx
            sy = 0.0 if abs(sy) < 1e-7 else sy
            if not any(np.hypot(sx - ux, sy - uy) < 1e-3 for ux, uy in unique_sols_2d):
                unique_sols_2d.append((sx, sy))
        unique_sols_2d.sort(key=lambda p: (round(p[0], 4), round(p[1], 4)))

        # Geometric Classification Banner
        st.markdown("### Geometric Diagnosis & Solutions")
        if is_linear_2d:
            A_mat, b_vec = sp.linear_eq_to_matrix([f1, f2], [x, y])
            rank_A = A_mat.rank()
            rank_Aug = A_mat.row_join(b_vec).rank()
            if rank_A == 2 and rank_Aug == 2 and unique_sols_2d:
                pt = unique_sols_2d[0]
                st.success(f"**Consistent & Independent (Rank = 2):** Two non-parallel lines intersecting at a single unique point **({pt[0]:.4f}, {pt[1]:.4f})**.")
            elif rank_A < rank_Aug:
                st.error("**Inconsistent System (Parallel Lines):** The lines have identical slopes but distinct intercepts — they never intersect (No Solution).")
            else:
                st.warning(f"**Consistent & Dependent (Coincident Lines):** Both equations represent the exact same line — infinitely many solutions. {symbolic_status}")
        else:
            if symbolic_status:
                st.warning(f"**Overlapping Curves:** {symbolic_status}")
            elif unique_sols_2d:
                st.success(f"**Found {len(unique_sols_2d)} Real Intersection Point(s):**")
            else:
                st.info("**No Real Intersections Found:** The two curves do not intersect in the real plane (or intersect outside numerical reach).")

        if unique_sols_2d:
            sol_cols = st.columns(min(len(unique_sols_2d), 4))
            for idx, (px, py) in enumerate(unique_sols_2d):
                with sol_cols[idx % 4]:
                    st.metric(label=f"Intersection #{idx+1}", value=f"({px:.4f}, {py:.4f})")

        # 2D Contour Plot of f1(x, y) = 0 and f2(x, y) = 0
        lin = np.linspace(-bound, bound, res_2d)
        X, Y = np.meshgrid(lin, lin)
        Z1 = f1_num(X, Y)
        Z2 = f2_num(X, Y)
        if np.isscalar(Z1):
            Z1 = np.full_like(X, Z1)
        if np.isscalar(Z2):
            Z2 = np.full_like(X, Z2)

        fig2d = go.Figure()

        fig2d.add_trace(go.Contour(
            x=lin, y=lin, z=Z1,
            contours=dict(start=0, end=0, size=1, coloring="none"),
            line=dict(color="#1E88E5", width=4),
            showscale=False,
            name=f"Eq 1: {eq1_str}"
        ))
        # Dummy legend entry because Plotly Contour coloring='none' hides legend items in some versions
        fig2d.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="#1E88E5", width=4),
            name=f"Eq 1: {eq1_str}"
        ))

        fig2d.add_trace(go.Contour(
            x=lin, y=lin, z=Z2,
            contours=dict(start=0, end=0, size=1, coloring="none"),
            line=dict(color="#E53935", width=4, dash="dash" if "Coincident" in preset_2d else "solid"),
            showscale=False,
            name=f"Eq 2: {eq2_str}"
        ))
        fig2d.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="#E53935", width=4),
            name=f"Eq 2: {eq2_str}"
        ))

        if unique_sols_2d:
            fig2d.add_trace(go.Scatter(
                x=[p[0] for p in unique_sols_2d],
                y=[p[1] for p in unique_sols_2d],
                mode="markers+text",
                text=[f"({p[0]:.2f}, {p[1]:.2f})" for p in unique_sols_2d],
                textposition="top right",
                textfont=dict(size=12, color="black"),
                marker=dict(size=12, color="#FFD700", symbol="diamond", line=dict(width=2, color="black")),
                name="Intersection Solution(s)"
            ))

        fig2d.update_layout(
            height=680,
            xaxis=dict(title="x", range=[-bound, bound], zeroline=True, zerolinewidth=1.5, zerolinecolor="gray", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="y", range=[-bound, bound], zeroline=True, zerolinewidth=1.5, zerolinecolor="gray"),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig2d, use_container_width=True)

    # --------------------------------------------------------------------------
    # 3 VARIABLES (3D SPACE)
    # --------------------------------------------------------------------------
    else:
        preset_3d = st.sidebar.selectbox(
            "Load Example System (3D):",
            options=[
                "Custom",
                "Planes: Unique Point Intersection",
                "Planes: Line of Intersection (Rank 2)",
                "Planes: Parallel Planes (No Solution)",
                "Planes: Triangular Prism (Pairwise Lines, No Common Point)",
                "Nonlinear: Sphere, Cylinder & Plane"
            ],
            index=1,
            key="preset_3d"
        )
        defaults_3d = {
            "Custom": (r"x + y + z = 3", r"2x - y + z = 2", r"x + 2y - z = 2"),
            "Planes: Unique Point Intersection": (r"x + y + z = 3", r"2x - y + z = 2", r"x + 2y - z = 2"),
            "Planes: Line of Intersection (Rank 2)": (r"x + y + z = 2", r"x - y + z = 0", r"2x + 2z = 2"),
            "Planes: Parallel Planes (No Solution)": (r"x + y + z = 1", r"x + y + z = -2", r"2x - y + z = 1"),
            "Planes: Triangular Prism (Pairwise Lines, No Common Point)": (r"x + y = 2", r"y + z = 2", r"x - z = 2"),
            "Nonlinear: Sphere, Cylinder & Plane": (r"x^2 + y^2 + z^2 = 9", r"x^2 + y^2 = 4", r"z = x + 1"),
        }
        d1, d2, d3 = defaults_3d[preset_3d]
        eq1_str = st.sidebar.text_input("Equation 1:", value=d1, key=f"eq3d_1_{preset_3d}")
        eq2_str = st.sidebar.text_input("Equation 2:", value=d2, key=f"eq3d_2_{preset_3d}")
        eq3_str = st.sidebar.text_input("Equation 3:", value=d3, key=f"eq3d_3_{preset_3d}")

        col_r1, col_r2 = st.sidebar.columns(2)
        with col_r1:
            bound3 = st.number_input("Box Range (±)", min_value=2.0, max_value=30.0, value=4.0, step=1.0, key="bound_3d")
        with col_r2:
            res3 = st.slider("3D Grid Resolution", min_value=20, max_value=50, value=32, step=4, key="res_3d")

        show_pairwise = st.sidebar.checkbox("Highlight Pairwise Intersection Curves (S₁∩S₂, S₂∩S₃, S₁∩S₃)", value=True)
        surf_opacity = st.sidebar.slider("Surface Opacity", min_value=0.15, max_value=0.85, value=0.45, step=0.05)

        try:
            f1 = parse_eq_to_expr(eq1_str, (x, y, z))
            f2 = parse_eq_to_expr(eq2_str, (x, y, z))
            f3 = parse_eq_to_expr(eq3_str, (x, y, z))
        except Exception as e:
            st.error(f"Equation parsing error: {e}")
            st.stop()

        st.sidebar.latex(f"\\text{{Eq 1: }} {sp.latex(f1)} = 0")
        st.sidebar.latex(f"\\text{{Eq 2: }} {sp.latex(f2)} = 0")
        st.sidebar.latex(f"\\text{{Eq 3: }} {sp.latex(f3)} = 0")

        f1_num = sp.lambdify((x, y, z), f1, modules=["numpy"])
        f2_num = sp.lambdify((x, y, z), f2, modules=["numpy"])
        f3_num = sp.lambdify((x, y, z), f3, modules=["numpy"])

        is_linear_3d = (
            f1.is_polynomial(x, y, z)
            and f2.is_polynomial(x, y, z)
            and f3.is_polynomial(x, y, z)
            and sp.Poly(f1, x, y, z).total_degree() <= 1
            and sp.Poly(f2, x, y, z).total_degree() <= 1
            and sp.Poly(f3, x, y, z).total_degree() <= 1
        )

        # Solve 3D system symbolically + numerically
        solutions_3d = []
        param_desc = ""
        try:
            sym_sols3 = sp.solve([f1, f2, f3], (x, y, z), dict=True)
            for s in sym_sols3:
                if x in s and y in s and z in s:
                    if not (s[x].free_symbols or s[y].free_symbols or s[z].free_symbols):
                        xv = complex(s[x].evalf())
                        yv = complex(s[y].evalf())
                        zv = complex(s[z].evalf())
                        if abs(xv.imag) < 1e-7 and abs(yv.imag) < 1e-7 and abs(zv.imag) < 1e-7:
                            solutions_3d.append((float(xv.real), float(yv.real), float(zv.real)))
                    else:
                        param_desc = f"$x = {sp.latex(s[x])},\\quad y = {sp.latex(s[y])},\\quad z = {sp.latex(s[z])}$"
        except Exception:
            pass

        def res_3d_fn(pt):
            try:
                v1 = float(f1_num(pt[0], pt[1], pt[2]))
                v2 = float(f2_num(pt[0], pt[1], pt[2]))
                v3 = float(f3_num(pt[0], pt[1], pt[2]))
                if not all(np.isfinite([v1, v2, v3])):
                    return np.array([1e6, 1e6, 1e6])
                return np.array([v1, v2, v3], dtype=float)
            except Exception:
                return np.array([1e6, 1e6, 1e6])

        seeds3 = np.linspace(-bound3, bound3, 5)
        for sx in seeds3:
            for sy in seeds3:
                for sz in seeds3:
                    r = root(res_3d_fn, x0=np.array([sx, sy, sz]), method="hybr", tol=1e-10)
                    if r.success and np.linalg.norm(res_3d_fn(r.x)) < 1e-6:
                        solutions_3d.append((float(r.x[0]), float(r.x[1]), float(r.x[2])))

        unique_sols_3d = []
        for sx, sy, sz in solutions_3d:
            sx = 0.0 if abs(sx) < 1e-7 else sx
            sy = 0.0 if abs(sy) < 1e-7 else sy
            sz = 0.0 if abs(sz) < 1e-7 else sz
            if not any(np.linalg.norm([sx - ux, sy - uy, sz - uz]) < 1e-3 for ux, uy, uz in unique_sols_3d):
                unique_sols_3d.append((sx, sy, sz))
        unique_sols_3d.sort(key=lambda p: (round(p[0], 4), round(p[1], 4), round(p[2], 4)))

        # Geometric Diagnosis
        st.markdown("### Geometric Diagnosis & Solutions")
        if is_linear_3d:
            A3, b3 = sp.linear_eq_to_matrix([f1, f2, f3], [x, y, z])
            rank_A3 = A3.rank()
            rank_Aug3 = A3.row_join(b3).rank()
            if rank_A3 == 3 and rank_Aug3 == 3 and unique_sols_3d:
                pt = unique_sols_3d[0]
                st.success(f"**Unique Point Intersection (Rank = 3):** All three planes meet at a single point **({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})**.")
            elif rank_A3 == 2 and rank_Aug3 == 2:
                st.warning(f"**Line of Intersection (Rank = 2):** All three planes share a common line of infinitely many solutions: {param_desc}")
            elif rank_A3 == 1 and rank_Aug3 == 1:
                st.warning(f"**Coincident Planes (Rank = 1):** All three equations describe the same plane: {param_desc}")
            else:
                st.error(f"**Inconsistent System (Rank(A)={rank_A3} < Rank([A|b])={rank_Aug3}):** At least two planes are parallel, or the three planes form a triangular prism with no common intersection point.")
        else:
            if unique_sols_3d:
                st.success(f"**Found {len(unique_sols_3d)} Real 3-Surface Intersection Point(s):**")
            elif param_desc:
                st.warning(f"**Curve/Surface of Intersection:** {param_desc}")
            else:
                st.info("**No Common 3-Surface Intersection Point Found** in real space.")

        if unique_sols_3d:
            cols3 = st.columns(min(len(unique_sols_3d), 4))
            for idx, (px, py, pz) in enumerate(unique_sols_3d):
                with cols3[idx % 4]:
                    st.metric(label=f"Solution #{idx+1}", value=f"({px:.3f}, {py:.3f}, {pz:.3f})")

        # Build 3D Plotly Scene
        fig3d = go.Figure()
        lin3 = np.linspace(-bound3, bound3, res3)
        X3, Y3, Z3 = np.meshgrid(lin3, lin3, lin3, indexing="ij")

        surface_specs = [
            (f1, f1_num, f"Eq 1: {eq1_str}", "#1E88E5"),
            (f2, f2_num, f"Eq 2: {eq2_str}", "#E53935"),
            (f3, f3_num, f"Eq 3: {eq3_str}", "#43A047"),
        ]

        # Helper to render an implicit surface F(x, y, z) = 0 cleanly
        def add_implicit_surface_trace(expr_sym, fn_num, label, color_hex):
            # Fast exact planar/explicit mesh if linear in z, y, or x
            for dep_var, order in [(z, ("x", "y")), (y, ("x", "z")), (x, ("y", "z"))]:
                if sp.degree(expr_sym, dep_var) == 1:
                    sol_dep = sp.solve(expr_sym, dep_var)
                    if sol_dep:
                        dep_expr = sol_dep[0]
                        if dep_var == z:
                            g_fn = sp.lambdify((x, y), dep_expr, modules=["numpy"])
                            Xg, Yg = np.meshgrid(lin3, lin3)
                            Zg = g_fn(Xg, Yg)
                            if np.isscalar(Zg):
                                Zg = np.full_like(Xg, Zg)
                            Zg = np.where((Zg >= -bound3) & (Zg <= bound3), Zg, np.nan)
                            fig3d.add_trace(go.Surface(
                                x=Xg, y=Yg, z=Zg,
                                colorscale=[[0, color_hex], [1, color_hex]],
                                opacity=surf_opacity,
                                showscale=False,
                                showlegend=True,
                                name=label
                            ))
                            return
                        elif dep_var == y:
                            g_fn = sp.lambdify((x, z), dep_expr, modules=["numpy"])
                            Xg, Zg = np.meshgrid(lin3, lin3)
                            Yg = g_fn(Xg, Zg)
                            if np.isscalar(Yg):
                                Yg = np.full_like(Xg, Yg)
                            Yg = np.where((Yg >= -bound3) & (Yg <= bound3), Yg, np.nan)
                            fig3d.add_trace(go.Surface(
                                x=Xg, y=Yg, z=Zg,
                                colorscale=[[0, color_hex], [1, color_hex]],
                                opacity=surf_opacity,
                                showscale=False,
                                showlegend=True,
                                name=label
                            ))
                            return
                        elif dep_var == x:
                            g_fn = sp.lambdify((y, z), dep_expr, modules=["numpy"])
                            Yg, Zg = np.meshgrid(lin3, lin3)
                            Xg = g_fn(Yg, Zg)
                            if np.isscalar(Xg):
                                Xg = np.full_like(Yg, Xg)
                            Xg = np.where((Xg >= -bound3) & (Xg <= bound3), Xg, np.nan)
                            fig3d.add_trace(go.Surface(
                                x=Xg, y=Yg, z=Zg,
                                colorscale=[[0, color_hex], [1, color_hex]],
                                opacity=surf_opacity,
                                showscale=False,
                                showlegend=True,
                                name=label
                            ))
                            return

            # General implicit 3D Isosurface for spheres, cylinders, hyperboloids, etc.
            V = fn_num(X3, Y3, Z3)
            if np.isscalar(V):
                V = np.full_like(X3, V)
            fig3d.add_trace(go.Isosurface(
                x=X3.flatten(),
                y=Y3.flatten(),
                z=Z3.flatten(),
                value=V.flatten(),
                isomin=0.0,
                isomax=0.0,
                surface_count=1,
                colorscale=[[0, color_hex], [1, color_hex]],
                opacity=surf_opacity,
                showscale=False,
                showlegend=True,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name=label
            ))

        for expr_i, fn_i, lbl_i, col_i in surface_specs:
            add_implicit_surface_trace(expr_i, fn_i, lbl_i, col_i)

        # Pairwise Intersection Curves (where two surfaces meet)
        if show_pairwise:
            pairs = [
                (f1, f2, f1_num, f2_num, "Eq 1 ∩ Eq 2", "#00E5FF"),
                (f2, f3, f2_num, f3_num, "Eq 2 ∩ Eq 3", "#FF9100"),
                (f1, f3, f1_num, f3_num, "Eq 1 ∩ Eq 3", "#D500F9"),
            ]
            t_grid = np.linspace(-bound3, bound3, 90)
            for fa, fb, fna, fnb, pair_lbl, p_col in pairs:
                # Try solving pairwise symbolically in terms of one free coordinate (x, y, or z)
                plotted_pair = False
                for free_var, dep_vars in [(z, (x, y)), (x, (y, z)), (y, (x, z))]:
                    try:
                        psol = sp.solve([fa, fb], dep_vars, dict=True)
                        if psol:
                            for branch in psol:
                                if all(dv in branch for dv in dep_vars):
                                    d1_fn = sp.lambdify(free_var, branch[dep_vars[0]], modules=["numpy"])
                                    d2_fn = sp.lambdify(free_var, branch[dep_vars[1]], modules=["numpy"])
                                    with np.errstate(all="ignore"):
                                        v1 = np.array(d1_fn(t_grid), dtype=complex)
                                        v2 = np.array(d2_fn(t_grid), dtype=complex)
                                    if v1.ndim == 0:
                                        v1 = np.full_like(t_grid, v1, dtype=complex)
                                    if v2.ndim == 0:
                                        v2 = np.full_like(t_grid, v2, dtype=complex)

                                    valid = (
                                        (np.abs(v1.imag) < 1e-6)
                                        & (np.abs(v2.imag) < 1e-6)
                                        & np.isfinite(v1.real)
                                        & np.isfinite(v2.real)
                                        & (np.abs(v1.real) <= bound3)
                                        & (np.abs(v2.real) <= bound3)
                                    )
                                    c1 = np.where(valid, v1.real, np.nan)
                                    c2 = np.where(valid, v2.real, np.nan)
                                    if np.any(valid):
                                        coords = {free_var: t_grid, dep_vars[0]: c1, dep_vars[1]: c2}
                                        fig3d.add_trace(go.Scatter3d(
                                            x=coords[x], y=coords[y], z=coords[z],
                                            mode="lines",
                                            line=dict(color=p_col, width=6),
                                            name=pair_lbl
                                        ))
                                        plotted_pair = True
                            if plotted_pair:
                                break
                    except Exception:
                        continue

        # Plot 3-Surface Intersection Point(s)
        if unique_sols_3d:
            fig3d.add_trace(go.Scatter3d(
                x=[p[0] for p in unique_sols_3d],
                y=[p[1] for p in unique_sols_3d],
                z=[p[2] for p in unique_sols_3d],
                mode="markers+text",
                text=[f"Solution<br>({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for p in unique_sols_3d],
                textposition="top center",
                textfont=dict(size=11, color="black"),
                marker=dict(size=10, color="#FFD700", symbol="diamond", line=dict(width=2, color="black")),
                name="3-Surface Intersection Solution"
            ))

        fig3d.update_layout(
            height=740,
            margin=dict(l=0, r=0, b=0, t=30),
            uirevision="sys3d_constant",
            scene=dict(
                xaxis=dict(title="X", range=[-bound3, bound3]),
                yaxis=dict(title="Y", range=[-bound3, bound3]),
                zaxis=dict(title="Z", range=[-bound3, bound3]),
                aspectmode="cube",
                uirevision="sys3d_constant"
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.80)")
        )
        st.plotly_chart(fig3d, use_container_width=True)


# ==============================================================================
# TOP-LEVEL NAVIGATION ROUTER
# ==============================================================================
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Go to Page:",
    options=[
        "1. Partial, Directional & Lagrange",
        "2. Geometric System Solver (2D & 3D)"
    ],
    index=0
)
st.sidebar.markdown("---")

if selected_page == "1. Partial, Directional & Lagrange":
    page_derivatives_and_lagrange()
else:
    page_geometric_system_solver()

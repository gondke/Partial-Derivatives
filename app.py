import streamlit as st
import numpy as np
from scipy.optimize import root
import sympy as sp
from sympy.parsing.latex import parse_latex
import plotly.graph_objects as go

st.set_page_config(page_title="Lagrange Multipliers Explorer", layout="wide")

st.title("Constrained Optimization: Method of Lagrange Multipliers")
st.markdown(
    "Optimize an objective function $f$ subject to equality constraint(s) $g_i = 0$ by solving "
    "$\\nabla f = \\sum_{i=1}^{m} \\lambda_i \\nabla g_i$ and classifying critical points via the **Bordered Hessian Matrix**."
)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Problem Dimensions")
num_vars = st.sidebar.radio(
    "Number of Variables (n):",
    options=[2, 3],
    index=0,
    format_func=lambda n: f"{n} Variables ({', '.join(['x', 'y', 'z'][:n])})"
)

max_constraints = 1 if num_vars == 2 else 2
if max_constraints == 1:
    num_constraints = 1
    st.sidebar.info("With 2 variables (x, y), 1 equality constraint g(x, y) = 0 is supported.")
else:
    num_constraints = st.sidebar.radio(
        "Number of Constraints (m):",
        options=[1, 2],
        index=0,
        format_func=lambda m: f"{m} Constraint{'s' if m > 1 else ''}"
    )

st.sidebar.header("2. Objective & Constraints (LaTeX)")
if num_vars == 2:
    default_f = r"x^2 + 2y^2"
    default_g1 = r"x^2 + y^2 - 1"
    default_g2 = r"x + y - 1"
else:
    default_f = r"x + 2y + 3z"
    default_g1 = r"x^2 + y^2 + z^2 - 14"
    default_g2 = r"x + y + z - 6"

obj_str = st.sidebar.text_input(
    f"Objective function f({', '.join(['x', 'y', 'z'][:num_vars])}):",
    value=default_f,
    help="Enter LaTeX or standard Python math expression."
)

g1_str = st.sidebar.text_input(
    "Constraint 1 (g₁ = 0 or lhs = rhs):",
    value=default_g1,
    help="If you enter 'x^2 + y^2 = 1', it is automatically rewritten as 'x^2 + y^2 - 1 = 0'."
)

g2_str = ""
if num_constraints == 2:
    g2_str = st.sidebar.text_input(
        "Constraint 2 (g₂ = 0 or lhs = rhs):",
        value=default_g2
    )

col_g1, col_g2 = st.sidebar.columns(2)
with col_g1:
    grid_range = st.number_input("Plot Range (±)", min_value=1.0, max_value=25.0, value=4.0, step=0.5)
with col_g2:
    grid_res = st.slider("Plot Resolution", min_value=25, max_value=90, value=50, step=5)

# --- LATEX / SYMPY PARSER ---
x, y, z = sp.symbols('x y z')
lam1, lam2 = sp.symbols('lambda_1 lambda_2')
var_syms = [x, y] if num_vars == 2 else [x, y, z]
lam_syms = [lam1] if num_constraints == 1 else [lam1, lam2]

def parse_math_expr(raw_text: str, is_constraint: bool = False):
    """Parses LaTeX or standard SymPy syntax. For constraints, converts 'lhs = rhs' to 'lhs - rhs'."""
    cleaned = raw_text.strip()
    if is_constraint and "=" in cleaned:
        lhs_str, rhs_str = cleaned.split("=", 1)
        lhs_expr = _parse_single(lhs_str)
        rhs_expr = _parse_single(rhs_str)
        return sp.simplify(lhs_expr - rhs_expr)
    if not is_constraint and "=" in cleaned:
        cleaned = cleaned.split("=", 1)[1].strip()
    return _parse_single(cleaned)

def _parse_single(text_part: str):
    text_part = text_part.strip()
    try:
        parsed = parse_latex(text_part)
        subs_map = {}
        for sym in parsed.free_symbols:
            if sym.name == 'x':
                subs_map[sym] = x
            elif sym.name == 'y':
                subs_map[sym] = y
            elif sym.name == 'z':
                subs_map[sym] = z
            elif sym.name == 'e':
                subs_map[sym] = sp.E
        return parsed.subs(subs_map)
    except Exception:
        return sp.sympify(text_part)

try:
    f_expr = parse_math_expr(obj_str, is_constraint=False)
    g_exprs = [parse_math_expr(g1_str, is_constraint=True)]
    if num_constraints == 2:
        g_exprs.append(parse_math_expr(g2_str, is_constraint=True))
except Exception as e:
    st.error(f"Could not parse mathematical input: {e}")
    st.stop()

# Build Lagrangian: L = f - sum(lambda_i * g_i)
L_expr = f_expr - sum(lam_syms[i] * g_exprs[i] for i in range(num_constraints))

# Display Problem Formulation
st.markdown("### 1. Lagrangian Formulation & First-Order Conditions")
col_form1, col_form2 = st.columns([1.2, 1])

with col_form1:
    st.latex(f"\\text{{Optimize: }} f({', '.join(str(v) for v in var_syms)}) = {sp.latex(f_expr)}")
    for idx, g_ex in enumerate(g_exprs):
        st.latex(f"\\text{{Subject to: }} g_{idx+1} = {sp.latex(g_ex)} = 0")
    st.latex(f"\\mathcal{{L}} = {sp.latex(L_expr)}")

with col_form2:
    st.markdown("**Stationarity & Constraint Equations ($\\nabla \\mathcal{L} = \\mathbf{0}$):**")
    grad_eqs = [sp.diff(L_expr, v) for v in var_syms] + [-g for g in g_exprs]
    eq_lines = [f"{sp.latex(sp.diff(L_expr, v))} = 0" for v in var_syms]
    for idx, g_ex in enumerate(g_exprs):
        eq_lines.append(f"{sp.latex(g_ex)} = 0")
    st.latex(r"\begin{cases} " + r" \\ ".join(eq_lines) + r" \end{cases}")

# --- BORDERED HESSIAN MATRIX SYMBOLIC CONSTRUCTION ---
# Structure:
# [  0_{m x m}      J_g      ]
# [   J_g^T      Hess_x(L)   ]
# where J_g is m x n Jacobian of constraints, Hess_x(L) is n x n Hessian of L wrt x
m = num_constraints
n = num_vars

zero_block = sp.zeros(m, m)
jac_g = sp.Matrix([[sp.diff(g_exprs[i], var_syms[j]) for j in range(n)] for i in range(m)])
hess_L_x = sp.Matrix([[sp.diff(L_expr, var_syms[i], var_syms[j]) for j in range(n)] for i in range(n)])

bordered_hess_sym = sp.BlockMatrix([[zero_block, jac_g], [jac_g.T, hess_L_x]]).as_explicit()

# Lambdify functions for numerical evaluation
all_unknowns = var_syms + lam_syms
f_num = sp.lambdify(var_syms, f_expr, modules=['numpy'])
g_nums = [sp.lambdify(var_syms, g, modules=['numpy']) for g in g_exprs]
grad_L_num = sp.lambdify(all_unknowns, grad_eqs, modules=['numpy'])
bordered_hess_num = sp.lambdify(all_unknowns, bordered_hess_sym, modules=['numpy'])

# --- CRITICAL POINT SOLVER (SYMBOLIC + MULTI-START NUMERICAL) ---
def solve_constrained_critical_points(bound: float):
    raw_solutions = []

    # 1. Exact Symbolic Solve
    try:
        sols = sp.solve(grad_eqs, all_unknowns, dict=True)
        for s in sols:
            if all(u in s for u in all_unknowns):
                vals = [complex(s[u].evalf()) for u in all_unknowns]
                if all(abs(v.imag) < 1e-7 for v in vals):
                    raw_solutions.append([float(v.real) for v in vals])
    except Exception:
        pass

    # 2. Multi-start Numerical Root Finding (catches transcendental systems)
    def residual_fn(vec):
        try:
            res_vals = np.array(grad_L_num(*vec), dtype=float)
            if not np.all(np.isfinite(res_vals)):
                return np.full(len(all_unknowns), 1e6)
            return res_vals
        except Exception:
            return np.full(len(all_unknowns), 1e6)

    grid_1d = np.linspace(-bound, bound, 7 if n == 2 else 5)
    lam_seeds = [-1.0, 1.0]
    if n == 2:
        seeds = [[vx, vy, l1] for vx in grid_1d for vy in grid_1d for l1 in lam_seeds]
    else:
        if m == 1:
            seeds = [[vx, vy, vz, l1] for vx in grid_1d for vy in grid_1d for vz in grid_1d for l1 in [1.0]]
        else:
            seeds = [[vx, vy, vz, 1.0, 1.0] for vx in grid_1d for vy in grid_1d for vz in grid_1d]

    for seed in seeds:
        sol = root(residual_fn, x0=np.array(seed, dtype=float), method='hybr', tol=1e-10)
        if sol.success and np.linalg.norm(residual_fn(sol.x)) < 1e-6:
            raw_solutions.append([float(val) for val in sol.x])

    # Deduplicate and filter to plot bounds
    unique_sols = []
    for sol_vec in raw_solutions:
        coords = sol_vec[:n]
        if all(-bound - 1e-4 <= c <= bound + 1e-4 for c in coords):
            cleaned = [0.0 if abs(c) < 1e-7 else c for c in sol_vec]
            if not any(np.linalg.norm(np.array(cleaned[:n]) - np.array(u[:n])) < 1e-3 for u in unique_sols):
                unique_sols.append(cleaned)

    unique_sols.sort(key=lambda s: tuple(round(c, 4) for c in s[:n]))
    return unique_sols

def classify_with_bordered_hessian(sol_vec):
    """Classifies constrained critical points using leading principal minors of the Bordered Hessian."""
    H_bar = np.array(bordered_hess_num(*sol_vec), dtype=float)
    f_val = float(f_num(*sol_vec[:n]))

    # Leading principal minors |H_k| for k = 2m + 1 up to n + m
    # Number of minors to check is n - m
    minors = []
    for k in range(2 * m + 1, n + m + 1):
        sub_mat = H_bar[:k, :k]
        minors.append((k, float(np.linalg.det(sub_mat))))

    # Classification rule for m equality constraints and n variables:
    # - Local Minimum: (-1)^m * |H_k| > 0 for all k = 2m+1, ..., n+m
    # - Local Maximum: (-1)^(k - m) * |H_k| > 0 for all k = 2m+1, ..., n+m (alternating signs starting with (-1)^(m+1))
    tol = 1e-6
    if any(abs(det_val) <= tol for _, det_val in minors):
        classification = "Indeterminate (Singular Minor)"
    else:
        is_min = all(((-1) ** m) * det_val > tol for _, det_val in minors)
        is_max = all(((-1) ** (k - m)) * det_val > tol for k, det_val in minors)
        if is_min:
            classification = "Constrained Minimum"
        elif is_max:
            classification = "Constrained Maximum"
        else:
            classification = "Constrained Saddle Point"

    return {
        "coords": sol_vec[:n],
        "lambdas": sol_vec[n:],
        "f_val": f_val,
        "H_bar": H_bar,
        "minors": minors,
        "classification": classification
    }

solutions = solve_constrained_critical_points(grid_range)
classified_results = [classify_with_bordered_hessian(s) for s in solutions]

# --- SECTION 2: CRITICAL POINTS & BORDERED HESSIAN ANALYSIS ---
st.markdown("### 2. Critical Points & Bordered Hessian Second-Derivative Test")

if not classified_results:
    st.warning("No constrained critical points found inside the current plot range. Try increasing the Plot Range.")
else:
    summary_rows = []
    for idx, res in enumerate(classified_results):
        row = {"#": idx + 1, "Classification": res["classification"]}
        for v_idx, v_sym in enumerate(var_syms):
            row[str(v_sym)] = round(res["coords"][v_idx], 4)
        row["f(x*)"] = round(res["f_val"], 4)
        for l_idx in range(m):
            row[f"λ_{l_idx+1}"] = round(res["lambdas"][l_idx], 4)
        for k, det_val in res["minors"]:
            row[f"|H̄_{k}|"] = round(det_val, 4)
        summary_rows.append(row)

    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    # Detailed Bordered Hessian Inspector for a chosen critical point
    selected_idx = st.selectbox(
        "Inspect Bordered Hessian Matrix & Principal Minors at Critical Point:",
        options=range(len(classified_results)),
        format_func=lambda i: (
            f"Point #{i+1}: {classified_results[i]['classification']} at "
            f"({', '.join(f'{c:.3f}' for c in classified_results[i]['coords'])}) "
            f"→ f = {classified_results[i]['f_val']:.4f}"
        )
    )

    chosen = classified_results[selected_idx]
    h_col1, h_col2 = st.columns([1.2, 1])

    with h_col1:
        st.markdown("**Symbolic Bordered Hessian $\\bar{H}$:**")
        st.latex(r"\bar{H} = " + sp.latex(bordered_hess_sym))

    with h_col2:
        st.markdown(f"**Evaluated $\\bar{{H}}$ at Point #{selected_idx + 1}:**")
        eval_mat_sym = sp.Matrix(np.round(chosen["H_bar"], 4))
        st.latex(r"\bar{H}(P^*) = " + sp.latex(eval_mat_sym))

        minor_str = ", \\quad ".join([f"|\\bar{{H}}_{{{k}}}| = {det_v:.4f}" for k, det_v in chosen["minors"]])
        st.latex(minor_str)

        if chosen["classification"] == "Constrained Minimum":
            st.success(f"**Conclusion:** $(-1)^m |\\bar{{H}}_k| > 0$ for $k = {2*m+1} \\dots {n+m}$ $\\implies$ **{chosen['classification']}** ($f = {chosen['f_val']:.4f}$).")
        elif chosen["classification"] == "Constrained Maximum":
            st.success(f"**Conclusion:** $(-1)^{{k-m}} |\\bar{{H}}_k| > 0$ for $k = {2*m+1} \\dots {n+m}$ $\\implies$ **{chosen['classification']}** ($f = {chosen['f_val']:.4f}$).")
        else:
            st.warning(f"**Conclusion:** Leading principal minors violate definite sign conditions $\\implies$ **{chosen['classification']}**.")

# --- SECTION 3: 3D GRAPHICAL PRESENTATION ---
st.markdown("### 3. 3D Graphical Visualization")

fig = go.Figure()

if num_vars == 2:
    # 2 Variables: Plot Surface z = f(x, y), Constraint g(x, y) = 0 on xy-plane,
    # and the Constrained Path (x, y, f(x, y)) directly over g(x, y) = 0 in contrasting color!
    lin = np.linspace(-grid_range, grid_range, grid_res)
    X, Y = np.meshgrid(lin, lin)
    Z = f_num(X, Y)
    if np.isscalar(Z):
        Z = np.full_like(X, Z)
    G1 = g_nums[0](X, Y)
    if np.isscalar(G1):
        G1 = np.full_like(X, G1)

    # 1. Objective Surface z = f(x, y)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        opacity=0.68,
        name='Objective Surface z = f(x, y)',
        colorbar=dict(title='f(x, y)', x=1.02)
    ))

    # Extract 0-contour segments of g1(x, y) = 0 using a dense grid + matplotlib/contour extraction
    # Pure NumPy/Plotly approach via marching squares from contourpy (bundled with matplotlib) or grid zero-crossings:
    def extract_contour_segments(Xg, Yg, Gg):
        """Extracts line segments along Gg = 0 using simple marching squares."""
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
                edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
                for i1, i2 in edges:
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
    Gf = g_nums[0](Xf, Yf)
    if not np.isscalar(Gf):
        cx, cy = extract_contour_segments(Xf, Yf, Gf)
        if len(cx) > 0:
            # Evaluate z = f(x, y) directly along the constraint curve
            cz = []
            for px_val, py_val in zip(cx, cy):
                if px_val is None:
                    cz.append(None)
                else:
                    cz.append(float(f_num(px_val, py_val)))

            # 2. Part of function z = f(x, y) over constraint g(x, y) = 0 (highlighted in bright crimson)
            fig.add_trace(go.Scatter3d(
                x=cx, y=cy, z=cz,
                mode='lines',
                line=dict(color='#FF1744', width=9),
                name='Function over Constraint: f(x, y) on g(x, y) = 0'
            ))

            # 3. Constraint curve g(x, y) = 0 projected onto base plane (z = min(Z))
            z_base = float(np.nanmin(Z))
            cz_base = [None if px_val is None else z_base for px_val in cx]
            fig.add_trace(go.Scatter3d(
                x=cx, y=cy, z=cz_base,
                mode='lines',
                line=dict(color='#00E5FF', width=5, dash='dash'),
                name='Constraint Curve g(x, y) = 0 (projected)'
            ))

    # 4. Highlight Constrained Critical Points on the Surface
    style_map = {
        "Constrained Maximum": dict(color="#FFD700", symbol="diamond", size=10),
        "Constrained Minimum": dict(color="#00FF7F", symbol="diamond", size=10),
        "Constrained Saddle Point": dict(color="#FF6D00", symbol="cross", size=10),
        "Indeterminate (Singular Minor)": dict(color="#FFFFFF", symbol="square", size=9),
    }
    for c_type, st_cfg in style_map.items():
        pts = [r for r in classified_results if r["classification"] == c_type]
        if pts:
            fig.add_trace(go.Scatter3d(
                x=[p["coords"][0] for p in pts],
                y=[p["coords"][1] for p in pts],
                z=[p["f_val"] for p in pts],
                mode='markers+text',
                text=[f"{c_type}<br>({p['coords'][0]:.2f}, {p['coords'][1]:.2f}, {p['f_val']:.2f})" for p in pts],
                textposition="top center",
                textfont=dict(size=11, color="black"),
                marker=dict(
                    size=st_cfg["size"],
                    color=st_cfg["color"],
                    symbol=st_cfg["symbol"],
                    line=dict(width=2, color="black")
                ),
                name=f"{c_type} ({len(pts)})"
            ))

    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, b=0, t=30),
        uirevision='lagrange_lock',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z = f(x, y)',
            aspectmode='cube',
            uirevision='lagrange_lock'
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.82)')
    )

else:
    # 3 Variables: Plot Isosurface g1(x, y, z) = 0 colored by objective f(x, y, z)
    lin3 = np.linspace(-grid_range, grid_range, min(grid_res, 40))
    X3, Y3, Z3 = np.meshgrid(lin3, lin3, lin3, indexing='ij')
    G1_3 = g_nums[0](X3, Y3, Z3)
    F_3 = f_num(X3, Y3, Z3)
    if np.isscalar(F_3):
        F_3 = np.full_like(X3, F_3)

    fig.add_trace(go.Isosurface(
        x=X3.flatten(),
        y=Y3.flatten(),
        z=Z3.flatten(),
        value=G1_3.flatten(),
        isomin=0,
        isomax=0,
        surface_count=1,
        opacity=0.55,
        colorscale='Plasma',
        name='Constraint Surface g₁(x, y, z) = 0',
        showscale=False
    ))

    if num_constraints == 2:
        G2_3 = g_nums[1](X3, Y3, Z3)
        fig.add_trace(go.Isosurface(
            x=X3.flatten(),
            y=Y3.flatten(),
            z=Z3.flatten(),
            value=G2_3.flatten(),
            isomin=0,
            isomax=0,
            surface_count=1,
            opacity=0.50,
            colorscale='Viridis',
            name='Constraint Surface g₂(x, y, z) = 0',
            showscale=False
        ))

    # Highlight 3D Critical Points
    style_map_3d = {
        "Constrained Maximum": dict(color="#FFD700", symbol="diamond", size=10),
        "Constrained Minimum": dict(color="#00FF7F", symbol="diamond", size=10),
        "Constrained Saddle Point": dict(color="#FF1744", symbol="cross", size=10),
        "Indeterminate (Singular Minor)": dict(color="#FFFFFF", symbol="square", size=9),
    }
    for c_type, st_cfg in style_map_3d.items():
        pts = [r for r in classified_results if r["classification"] == c_type]
        if pts:
            fig.add_trace(go.Scatter3d(
                x=[p["coords"][0] for p in pts],
                y=[p["coords"][1] for p in pts],
                z=[p["coords"][2] for p in pts],
                mode='markers+text',
                text=[
                    f"{c_type}<br>({p['coords'][0]:.2f}, {p['coords'][1]:.2f}, {p['coords'][2]:.2f})<br>f = {p['f_val']:.2f}"
                    for p in pts
                ],
                textposition="top center",
                marker=dict(
                    size=st_cfg["size"],
                    color=st_cfg["color"],
                    symbol=st_cfg["symbol"],
                    line=dict(width=2, color="black")
                ),
                name=f"{c_type} ({len(pts)})"
            ))

    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, b=0, t=30),
        uirevision='lagrange_3d_lock',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
            uirevision='lagrange_3d_lock'
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.82)')
    )

st.plotly_chart(fig, use_container_width=True)

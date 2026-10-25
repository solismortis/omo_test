"""Реализовать решение задачи оптимизации траектории. Для начала можно рассмотреть ситуацию,
когда параметры ограничивающих кругов уже известны.

Differential drive. The origin is at the bottom left. Meters
"""

import math
import random

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
from matplotlib import pyplot as plt
import numpy as np


def export_diff_drive_model():
    model = AcadosModel()
    model.name = 'diff_drive'

    # States
    x = ca.SX.sym('x')  # SX means declare a scalar, sim() takes the name
    y = ca.SX.sym('y')
    v = ca.SX.sym('v')  # Speed
    theta = ca.SX.sym('theta')
    x_state = ca.vertcat(x, y, v, theta)  # State vector. Vertical concatenation (a vector, also considered a column matrix)

    # Controls
    a = ca.SX.sym('a')
    omega = ca.SX.sym('omega')
    u_ctrl = ca.vertcat(a, omega)  # Control vector

    # Explicit differential equations
    f_expl = ca.vertcat(v * ca.cos(theta),  # = dx/dt
                        v * ca.sin(theta),  # = dy/dt
                        a,  # = dv/dt
                        omega)  # = dtheta/dt

    model.x = x_state
    model.u = u_ctrl
    model.xdot = ca.SX.sym('xdot', 4)  # Dot in Newton's notation means time derivative. This here is a time der of the state vector
    model.f_expl_expr = f_expl
    model.f_impl_expr = model.xdot - f_expl  # Implicit. Acados likes having both

    return model


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class RefPath:
    """0 - start, -1 - finish"""
    def __init__(self, points: list[Point]):
        self.points = points


class Circle:
    def __init__(self, x0, y0, r):
        self.x0  = x0
        self.y0 = y0
        self.r = r


class Grid:
    def __init__(self, width, height, spacing):
        self.width = width
        self.height = height
        self.spacing = spacing


grid = Grid(10, 10, 0.1)
refpath = refpath = RefPath([
    Point(0.8, 0.8),
    Point(1.2, 0.8),
    Point(1.7, 0.82),
    Point(2.2, 0.9),
    Point(2.7, 1.05),
    Point(3.1, 1.25),
    Point(3.5, 1.55),
    Point(3.8, 1.9),
    Point(4.05, 2.3),
    Point(4.25, 2.75),
    Point(4.4, 3.25),
    Point(4.5, 3.8),
    Point(4.55, 4.35),
    Point(4.58, 4.9),
    Point(4.65, 5.45),
    Point(4.8, 6.0),
    Point(5.05, 6.5),
    Point(5.4, 6.95),
    Point(5.85, 7.35),
    Point(6.35, 7.7),
    Point(6.9, 7.95),
    Point(7.45, 8.15),
    Point(7.95, 8.35),
    Point(8.4, 8.6),
    Point(8.75, 8.9),
    Point(9.0, 9.2),
    Point(9.15, 9.45),
    Point(9.25, 9.65),
    Point(9.32, 9.8),
    Point(9.4, 9.9),
])

# Generate ref theta (robot orientation) as angles from a point to point
def ref_theta(points: list[Point]) -> list[float]:
    n = len(points)
    thetas = []
    for i in range(n):
        if i == 0:
            dx = points[1].x - points[0].x
            dy = points[1].y - points[0].y
        elif i == n - 1:
            dx = points[-1].x - points[-2].x
            dy = points[-1].y - points[-2].y
        else:
            dx = points[i + 1].x - points[i - 1].x
            dy = points[i + 1].y - points[i - 1].y
        thetas.append(math.atan2(dy, dx))
    return thetas

# v (speed) will be as a constant (can even calculate v if you want)
v_ref = [1] * len(refpath.points)

circles = []
for p in refpath.points:
    circles.append(Circle(p.x, p.y, 1))

ref_state = [[p.x for p in refpath.points],
             [p.y for p in refpath.points],
             v_ref,
             ref_theta(refpath.points)]
print(ref_state)


# N = 29  # prediction horizon (also called the horizon length or number of shooting intervals
# N has to be < number of circles (ref points). Otherwise, doesn't work. 
# Tf = 20.0  # total horizon time. Has to align with N
# Acados needs Tf and the NLP needs N, and the relationship Δt = Tf/N is the thing that actually matters
# Acados internally computes the step as Δt = Tf / N

N = len(refpath.points) - 1   # = 29, so N+1 == 30 reference points
Tf = 14.5

def resample_reference(ref_points, circles, n_steps):
    """Linearly resample the reference path (and its circles) to n_steps+1
    samples so that each OCP stage k gets a reference and a bounding circle.
    """
    # Arc-length parameterization of the reference polyline
    xs = np.array([p.x for p in ref_points], dtype=float)
    ys = np.array([p.y for p in ref_points], dtype=float)
    seg = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]

    s_target = np.linspace(0.0, total, n_steps + 1)

    x_ref = np.interp(s_target, s, xs)
    y_ref = np.interp(s_target, s, ys)

    # Resample theta and v from the precomputed lists in part1
    thetas = np.array(ref_theta(ref_points), dtype=float)
    v_arr = np.array(v_ref, dtype=float)
    th_ref = np.interp(s_target, s, thetas)
    v_ref_rs = np.interp(s_target, s, v_arr)

    # Resample circle centers/radii
    cx = np.array([c.x0 for c in circles], dtype=float)
    cy = np.array([c.y0 for c in circles], dtype=float)
    r = np.array([c.r for c in circles], dtype=float)
    cx_rs = np.interp(s_target, s, cx)
    cy_rs = np.interp(s_target, s, cy)
    r_rs = np.interp(s_target, s, r)

    # Pack into the (nx_ref, N+1) style array used later
    X_ref = np.vstack([x_ref, y_ref, v_ref_rs, th_ref])       # (4, N+1)
    circles_rs = np.vstack([cx_rs, cy_rs, r_rs])              # (3, N+1)
    return X_ref, circles_rs


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

x_start = np.array([0.8, 0.8, 0.0, 0.0])
x_goal = np.array([9.4, 9.9, 0.5, np.pi / 4])

model = export_diff_drive_model()
ocp = AcadosOcp()
ocp.model = model

nx = model.x.rows()
nu = model.u.rows()

# --- dims ------------------------------------------------------------
ocp.solver_options.N_horizon = N
np_param = 7           # [xref, yref, thref, vref, cx, cy, r]
ocp.dims.np = np_param
ocp.parameter_values = np.zeros(np_param)

# --- cost ------------------------------------------------------------
W_x = np.diag([10.0, 10.0, 1.0, 1.0])
W_u = np.diag([0.1, 0.1])
W_xe = np.diag([100.0, 100.0, 10.0, 10.0])

ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.cost_type_e = 'LINEAR_LS'
ocp.cost.W = np.block([[W_x, np.zeros((nx, nu))],
                        [np.zeros((nu, nx)), W_u]])
ocp.cost.W_e = W_xe

ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
ocp.cost.yref = np.zeros(nx + nu)
ocp.cost.yref_e = np.zeros(nx)
ocp.cost.Vx_e = np.eye(nx)

# parameters
p_sym = ca.SX.sym('p', np_param)
ocp.model.p = p_sym

# CIAO constraint (per-stage): (x-cx)^2 + (y-cy)^2 <= r^2
x_sym = ocp.model.x
dx = x_sym[0] - p_sym[4]
dy = x_sym[1] - p_sym[5]
ocp.model.con_h_expr = ca.vertcat(dx * dx + dy * dy - p_sym[6] * p_sym[6])
ocp.constraints.lh = np.array([-1e9])
ocp.constraints.uh = np.array([0.0])

# --- initial state --------------------------------------------------
ocp.constraints.x0 = x_start

# --- control bounds -------------------------------------------------
ocp.constraints.lbu = np.array([-2.0, -2.0])
ocp.constraints.ubu = np.array([2.0, 2.0])
ocp.constraints.idxbu = np.array([0, 1])

# --- terminal constraint (soft box around goal) ---------------------
xe = ocp.model.x
ocp.model.con_h_expr_e = ca.vertcat(
    xe[0] - x_goal[0],
    xe[1] - x_goal[1],
    xe[2] - x_goal[2],
    xe[3] - x_goal[3],
)
tol = np.array([1.0, 1.0, 0.5, 0.5])
ocp.constraints.lh_e = -tol
ocp.constraints.uh_e = tol

# --- solver options -------------------------------------------------
ocp.solver_options.tf = Tf
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.nlp_solver_max_iter = 200

ocp.code_gen_options.json_file = 'ciao_step1.json'

# --- references & circles (resampled from part1 data) ---------------
x_ref, circles_rs = resample_reference(refpath.points, circles, N)

# --- build solver ---------------------------------------------------
solver = AcadosOcpSolver(ocp, verbose=False)

# --- feed parameters, yrefs, initial guess --------------------------
for k in range(N):
    ref_k = x_ref[:, k]
    circle_k = circles_rs[:, k]
    p_val = np.array([ref_k[0], ref_k[1], ref_k[2], ref_k[3],
                        circle_k[0], circle_k[1], circle_k[2]])
    solver.set(k, 'p', p_val)
    solver.set(k, 'yref', np.concatenate([ref_k, np.zeros(nu)]))
solver.set(N, 'yref', x_goal)

for k in range(N + 1):
    solver.set(k, 'x', x_ref[:, k])
for k in range(N):
    solver.set(k, 'u', np.zeros(nu))

# --- solve ----------------------------------------------------------
status = solver.solve()
print(f'Solver status: {status} (0 = success)')

# --- extract --------------------------------------------------------
X = np.zeros((N + 1, nx))
U = np.zeros((N, nu))
for k in range(N + 1):
    X[k, :] = solver.get(k, 'x')
for k in range(N):
    U[k, :] = solver.get(k, 'u')

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))

# Grid boundary (from part1.Grid)
ax.set_xlim(-1, grid.width + 1)
ax.set_ylim(-1, grid.height + 1)

# Original circles (from part1)
for c in circles:
    ax.add_patch(plt.Circle((c.x0, c.y0), c.r, color='olive',
                            alpha=0.15, fill=True, lw=0.5))

# Original reference polyline (from part1.refpath)
xs_ref = [p.x for p in refpath.points]
ys_ref = [p.y for p in refpath.points]
ax.plot(xs_ref, ys_ref, 'k--', lw=1, label='Reference path')

# Optimized trajectory
ax.plot(X[:, 0], X[:, 1], 'g-', lw=2, label='CIAO (acados)')

ax.plot(x_start[0], x_start[1], 'bo', ms=8, label='Start')
ax.plot(x_goal[0], x_goal[1], 'ro', ms=8, label='Goal')

ax.set_aspect('equal')
ax.grid()
ax.legend()
plt.tight_layout()
plt.show()
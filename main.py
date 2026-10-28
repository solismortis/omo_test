"""Реализовать решение задачи оптимизации траектории. Для начала можно рассмотреть ситуацию,
когда параметры ограничивающих кругов уже известны.

Differential drive. The origin is at the bottom left. Meters
We basically ignore the CIAO* paper for now, as it mostly adds different norms (while L2 is the optimal one),
works with complex-form robots (too much trouble for now). We also do offline (not MPC), cause brain already hurts
(offline is for static env and we see the entire path to the goal, MPC is for IRL robot control).

Using A* for ref_path. For A* we reduce grid resolution and only use 4 movement directions to make sure the robot can fit
(otherwise, A* slips in between 2 cells adjacent diagonally like the robot is a dot)
"""

# TODO: Max 100 ms
# TODO: v being a const is not great

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
    def __init__(self, x, y, r):
        self.x  = x
        self.y = y
        self.r = r


class Grid:
    def __init__(self, width, height, spacing):
        self.width = width
        self.height = height
        self.spacing = spacing


grid = Grid(10, 10, 0.1)
# 30 hand-crafted points
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

x_ref = np.vstack([[p.x for p in refpath.points],
                   [p.y for p in refpath.points],
                   v_ref,
                   ref_theta(refpath.points)])

circles_rs = np.vstack([[c.x for c in circles],
                        [c.y for c in circles],
                        [c.r for c in circles]])

N = len(refpath.points) - 1  # prediction horizon (also called the horizon length or number of shooting intervals),
# has to be < number of circles. So far, the only workable N is a bit lower than number of points, can't go too low, can't go 1 more

Tf = 15  # total horizon time. Has to align with N
# Acados needs Tf and the NLP needs N, and the relationship Δt = Tf/N is the thing that actually matters
# Acados internally computes the step as Δt = Tf / N

# Main solver

x_start = np.array([x_ref[0][0], x_ref[1][0], x_ref[2][0], x_ref[3][0]])
x_goal = np.array([x_ref[0][-1], x_ref[1][-1], x_ref[2][-1], x_ref[3][-1]])

model = export_diff_drive_model()
ocp = AcadosOcp()  # Optimal control problem
ocp.model = model

# Dimensions
nx = model.x.rows()  # State vector
nu = model.u.rows()  # Control vector
ocp.solver_options.N_horizon = N
np_param = 7  # [xref, yref, thref, vref, cx, cy, r]
ocp.dims.np = np_param
ocp.parameter_values = np.zeros(np_param)

# Cost
# np.diag() creates a 2D matrix with specified values on the diagonal
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

# CIAO constraint: (x-cx)^2 + (y-cy)^2 <= r^2
x_sym = ocp.model.x
dx = x_sym[0] - p_sym[4]
dy = x_sym[1] - p_sym[5]
ocp.model.con_h_expr = ca.vertcat(dx * dx + dy * dy - p_sym[6] * p_sym[6])
ocp.constraints.lh = np.array([-1e9])
ocp.constraints.uh = np.array([0.0])

# Initial state
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

ocp.solver_options.tf = Tf
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP'
ocp.solver_options.nlp_solver_max_iter = 200

ocp.code_gen_options.json_file = 'ciao_step1.json'

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



# Visualization

fig, ax = plt.subplots(figsize=(7, 7))

ax.set_xlim(-1, grid.width + 1)
ax.set_ylim(-1, grid.height + 1)

for c in circles:
    ax.add_patch(plt.Circle((c.x, c.y), c.r, color='olive', alpha=0.15, fill=True, lw=0.5))

ax.plot(X[:, 0], X[:, 1], 'g-', lw=2, label='CIAO')

xs_ref = [p.x for p in refpath.points]
ys_ref = [p.y for p in refpath.points]
ax.plot(xs_ref, ys_ref, 'k--', lw=1, label='Reference path')

# Start and finish
ax.text(refpath.points[0].x ,refpath.points[0].y, 'S', color='blue', fontsize=16, fontweight='bold', ha='center', va='center')
ax.text(refpath.points[-1].x ,refpath.points[-1].y, 'G', color='blue', fontsize=16, fontweight='bold', ha='center', va='center')

ax.set_aspect('equal')
ax.grid()
ax.legend()
plt.tight_layout()
plt.show()
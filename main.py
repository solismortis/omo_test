"""
Differential drive. The origin is at the bottom left. We use meters
We basically ignore the CIAO* paper for now, as it mostly adds different norms (while L2 is the optimal one anyway),
works with complex-form robots (too much trouble for this task). We also do offline (not MPC), cause brain already hurts
(offline is for static env and we see the entire path to the goal (unless we truncate), MPC is for IRL robot control).

Using A* for ref_path. For A* we reduce grid resolution and only use 4 movement directions to make sure the robot can fit
(otherwise, A* slips in between 2 cells adjacent diagonally like the robot is a dot)

We receive a grid, put random obstacles and convert grid borders into wall obstacles,
then run A* and put intermediary points in between what A* returns (otherwise generated circles are not dense enough),
then we grow our circles (free balls), then we do Acados magic and plot everything.
"""

import math
import heapq
import random

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union


def heuristic(a, b):
    """Manhattan distance."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(grid, start, goal):
    """4 directions. If we try using 8, A* will slip in between obstacles like our robot is a dot"""
    # TODO: Make A* return "stairs" for diagonal paths. Will prob be better for CIAO
    rows = len(grid)
    cols = len(grid[0])

    def in_bounds(node):
        x, y = node
        return 0 < x < cols and 0 < y < rows

    def passable(node):
        x, y = node
        return grid[y][x] == 0

    def passable_neighbors(node):
        x, y = node
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in candidates if in_bounds(c) and passable(c)]

    if not (in_bounds(start) and in_bounds(goal)):
        return None
    if not (passable(start) and passable(goal)):
        return None

    open_heap = [(0.0, 0, start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    counter = 1

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in passable_neighbors(current):
            if neighbor in closed:
                continue
            tg = g_score[current] + 1
            if tg < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tg
                heapq.heappush(open_heap, (tg + heuristic(neighbor, goal), counter, neighbor))
                counter += 1

    return None


def random_grid(rows, cols, obstacle_prob=0.2):
    return [[1 if random.random() < obstacle_prob else 0
             for _ in range(cols)] for _ in range(rows)]


def densify_path(path, spacing):
    dense = [path[0]]
    for p, q in zip(path, path[1:]):
        px, py = p
        qx, qy = q
        seg_len = np.hypot(qx - px, qy - py)
        if seg_len == 0:
            continue
        n_steps = int(np.ceil(seg_len / spacing))
        for k in range(1, n_steps + 1):
            t = k / n_steps
            x = px + t * (qx - px)
            y = py + t * (qy - py)
            dense.append((x, y))
    return dense


def distance_to_nearest_obstacle(point, obstacle_centers):
    if not obstacle_centers:
        return float('inf')
    return min(np.hypot(point[0] - ox, point[1] - oy) for ox, oy in obstacle_centers)


def obstacle_gradient(point, obstacle_centers):
    """Normalized gradient of the distance function (points away from nearest obstacle)."""
    if not obstacle_centers:
        return np.array([0.0, 0.0])
    dists = [np.hypot(point[0] - ox, point[1] - oy) for ox, oy in obstacle_centers]
    i = int(np.argmin(dists))
    ox, oy = obstacle_centers[i]
    g = np.array([point[0] - ox, point[1] - oy], dtype=float)
    n = np.linalg.norm(g)
    if n < 1e-12:
        return np.array([0.0, 0.0])
    return g / n


def max_free_ball_center(path_point, obstacle_centers, obstacle_radius,
                         step_init=0.05, step_grow=1.6,
                         max_step=50.0, n_dirs=8, tol=1e-4):
    """Grow the maximal free ball around a path_point.

    Constraints enforced:
        radius(c) = dist(c, obstacles) - obstacle_radius  > 0
        ||c - path_point|| <= radius(c) (original center stays inside the ball)
    """
    base = np.array(path_point, dtype=float)

    def ball_radius(c):
        return distance_to_nearest_obstacle(c, obstacle_centers) - obstacle_radius

    def feasible(c):
        r = ball_radius(c)
        if r <= 0:
            return -1.0
        if np.linalg.norm(base - c) > r:
            return -1.0
        return r

    best_c = base.copy()
    best_r = ball_radius(base)
    if best_r <= 0:
        return best_c, 0.0

    primary = obstacle_gradient(base, obstacle_centers)
    if np.linalg.norm(primary) < 1e-9:
        return best_c, best_r

    dirs = [primary]
    ang0 = np.arctan2(primary[1], primary[0])
    for k in range(1, n_dirs):
        a = ang0 + 2 * np.pi * k / n_dirs
        dirs.append(np.array([np.cos(a), np.sin(a)]))

    for g in dirs:
        lo = 0.0
        hi = None
        step = step_init
        c = base.copy()
        last_feasible = base.copy()
        last_r = ball_radius(base)

        while step <= max_step:
            c_next = base + step * g
            r_next = feasible(c_next)
            if r_next <= 0:
                hi = step
                break
            lo = step
            last_feasible = c_next.copy()
            last_r = r_next
            step *= step_grow

        if hi is not None:
            while (hi - lo) > tol:
                mid = 0.5 * (lo + hi)
                c_mid = base + mid * g
                r_mid = feasible(c_mid)
                if r_mid > 0:
                    lo = mid
                    last_feasible = c_mid.copy()
                    last_r = r_mid
                else:
                    hi = mid

        if last_r > best_r:
            best_c, best_r = last_feasible.copy(), last_r

    return best_c, best_r


def grow_free_balls(path, obstacle_centers, obstacle_radius=0.71):
    balls = []
    for p in path:
        c, r = max_free_ball_center(p, obstacle_centers, obstacle_radius)
        balls.append((c, r))
    return balls


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


def ref_theta(points):
    """Computes, for every waypoint of the reference path, the direction of travel at that point and returns it as an angle in radians"""
    n = len(points)
    thetas = []
    for i in range(n):
        if i == 0:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
        elif i == n - 1:
            dx = points[-1][0] - points[-2][0]
            dy = points[-1][1] - points[-2][1]
        else:
            dx = points[i + 1][0] - points[i - 1][0]
            dy = points[i + 1][1] - points[i - 1][1]
        thetas.append(math.atan2(dy, dx))
    return thetas


def build_and_solve_ocp(x_ref, balls, x_start, x_goal, N, Tf, circle_safety_margin=0.0):
    model = export_diff_drive_model()
    ocp = AcadosOcp()
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
    W_x = np.diag([10.0, 10.0, 1.0, 1.0])  # [x, y, v, theta]
    W_u = np.diag([0.1, 0.1])  # [a, omega]

    ocp.cost.cost_type = 'LINEAR_LS'  # Linear least-squares cost function for the stage cost (every step except the terminal one)
    ocp.cost.W = np.block([[W_x, np.zeros((nx, nu))],
                           [np.zeros((nu, nx)), W_u]])

    ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
    ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
    ocp.cost.yref = np.zeros(nx + nu)

    p_sym = ca.SX.sym('p', np_param)
    ocp.model.p = p_sym

    x_sym = ocp.model.x
    dx = x_sym[0] - p_sym[4]
    dy = x_sym[1] - p_sym[5]
    r_safe = p_sym[6] - circle_safety_margin
    ocp.model.con_h_expr = ca.vertcat(dx * dx + dy * dy - r_safe * r_safe)
    ocp.constraints.lh = np.array([-1e9])
    ocp.constraints.uh = np.array([0.0])

    ocp.constraints.x0 = x_start

    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.code_gen_options.json_file = 'ciao_combined.json'

    solver = AcadosOcpSolver(ocp, verbose=False)

    for k in range(N):
        ref_k = x_ref[:, k]
        (cx, cy), r = balls[k]
        p_val = np.array([ref_k[0], ref_k[1], ref_k[2], ref_k[3],
                          cx, cy, r])
        solver.set(k, 'p', p_val)
        solver.set(k, 'yref', np.concatenate([ref_k, np.zeros(nu)]))

    for k in range(N + 1):
        solver.set(k, 'x', x_ref[:, k])
    for k in range(N):
        solver.set(k, 'u', np.zeros(nu))

    status = solver.solve()
    print(f'  Solver status: {status} (0 = success)')

    X = np.zeros((N + 1, nx))
    U = np.zeros((N, nu))
    for k in range(N + 1):
        X[k, :] = solver.get(k, 'x')
    for k in range(N):
        U[k, :] = solver.get(k, 'u')
    return X, U, status


# =============================================================================
# PART 3 — VISUALISATION
# =============================================================================

def plot_result(grid, start, goal, path_astar, densified, balls, X_opt=None,
                title="", obstacle_radius=0.71):
    rows, cols = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=(8, 8))

    grey_radius = 0.5
    red_circles = []
    obstacle_centers = []

    for y, row in enumerate(grid):
        for x, col in enumerate(row):
            if col == 1:
                ax.add_patch(plt.Circle((x, y), grey_radius, color='gray', zorder=1))
                obstacle_centers.append((x, y))
                red_circles.append(Point(x, y).buffer(obstacle_radius, resolution=32))

    wall_centers = []
    for x in range(cols + 1):
        wall_centers.append((float(x), 0.0))
        wall_centers.append((float(x), float(rows)))
    for y in range(1, rows):
        wall_centers.append((0.0, float(y)))
        wall_centers.append((float(cols), float(y)))
    for x, y in wall_centers:
        ax.add_patch(plt.Circle((x, y), grey_radius, color='gray', zorder=1))
        red_circles.append(Point(x, y).buffer(obstacle_radius, resolution=32))

    if red_circles:
        unified = unary_union(red_circles)
        polys = [unified] if unified.geom_type == 'Polygon' else list(unified.geoms)
        for poly in polys:
            ex_x, ex_y = poly.exterior.xy
            ax.plot(ex_x, ex_y, color='red', linewidth=1.5, zorder=0)
            for interior in poly.interiors:
                in_x, in_y = interior.xy
                ax.plot(in_x, in_y, color='red', linewidth=1.5, zorder=0)

    # free balls
    for p, (c, r) in zip(densified, balls):
        if r > 0.01:
            ax.add_patch(plt.Circle(c, r, fill=False, edgecolor=(0.592, 0.631, 0.043),
                                    linewidth=1.2, linestyle='--', zorder=2))
            if np.linalg.norm(np.array(c) - np.array(p)) > 0.02:
                ax.annotate('', xy=c, xytext=p,
                            arrowprops=dict(arrowstyle='->', color='purple', lw=0.6),
                            zorder=3)

    # A* path
    xs = [p[0] for p in path_astar]; ys = [p[1] for p in path_astar]
    ax.plot(xs, ys, '-', color=(0.282, 0.322, 0.329), lw=2, label='A*', zorder=4)
    ax.plot(xs, ys, 'o', color=(0.282, 0.322, 0.329), ms=3, zorder=4)

    # optimized trajectory
    if X_opt is not None:
        ax.plot(X_opt[:, 0], X_opt[:, 1], 'g-', lw=2.5, label='CIAO', zorder=5)

    ax.plot(*start, 'o', ms=10, mfc=(0.145, 0.678, 0.8), mec='black', label='Start', zorder=6)
    ax.plot(*goal, 'o', ms=10, mfc=(0.8, 0.145, 0.671), mec='black', label='Goal', zorder=6)

    ax.set_xlim(0, cols); ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()


# =============================================================================
# PART 4 — ONE FULL PIPELINE RUN
# =============================================================================

def main(seed=None, rows=10, cols=10, obstacle_prob=0.2,
                 spacing=0.33, obstacle_radius=0.71,
                 max_horizon=40, v_nominal=1.0):
    random.seed(seed)
    np.random.seed(seed)

    start = (1, 1)
    goal = (9, 9)
    grid = random_grid(rows, cols, obstacle_prob=obstacle_prob)
    grid[start[1]][start[0]] = 0
    grid[goal[1]][goal[0]] = 0

    path = a_star(grid, start, goal)
    if path is None:
        print(f'  seed={seed}: no A* path, skipping')
        return None

    densified = densify_path(path, spacing=spacing)

    # obstacles for free-ball growth: grid obstacles + wall pseudo-obstacles
    obstacle_centers = [(x, y) for y in range(rows) for x in range(cols)
                        if grid[y][x] == 1]
    for x in range(cols + 1):
        obstacle_centers.append((float(x), 0.0))
        obstacle_centers.append((float(x), float(rows)))
    for y in range(1, rows):
        obstacle_centers.append((0.0, float(y)))
        obstacle_centers.append((float(cols), float(y)))

    balls = grow_free_balls(densified, obstacle_centers, obstacle_radius=obstacle_radius)

    # ---- subsample to fit the OCP horizon --------------------------------
    N_total = len(densified) - 1
    N = min(N_total, max_horizon)
    if N < 4:
        print(f'  seed={seed}: path too short for OCP ({N_total} intervals), skipping')
        return None

    idx = np.linspace(0, len(densified) - 1, N + 1).round().astype(int)
    ref_pts = [densified[i] for i in idx]
    ref_balls = [balls[i] for i in idx]

    # reference speed: keep v * Tf / N ≈ spacing * (N_total / N) so it's consistent
    seg_lens = [np.hypot(ref_pts[i + 1][0] - ref_pts[i][0],
                         ref_pts[i + 1][1] - ref_pts[i][1]) for i in range(N)]
    Tf = max(5.0, sum(seg_lens) / v_nominal)

    x_ref = np.vstack([
        [p[0] for p in ref_pts],
        [p[1] for p in ref_pts],
        [v_nominal] * (N + 1),
        ref_theta(ref_pts),
    ])

    x_start = np.array([x_ref[0, 0], x_ref[1, 0], x_ref[2, 0], x_ref[3, 0]])
    x_goal = np.array([x_ref[0, -1], x_ref[1, -1], x_ref[2, -1], x_ref[3, -1]])

    X, U, status = build_and_solve_ocp(x_ref, ref_balls, x_start, x_goal,
                                       N, Tf, circle_safety_margin=0.0)

    plot_result(grid, start, goal, path, densified, balls, X_opt=X,
                title=f'seed={seed}, N={N}, status={status}',
                obstacle_radius=obstacle_radius)
    plt.show()

    return dict(grid=grid, path=path, densified=densified, balls=balls,
                X=X, U=U, status=status, N=N, Tf=Tf)


main(seed=None)
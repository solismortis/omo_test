import heapq
import random
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union


# 4 directions. 0 is free, 1 is an obstacle

def heuristic(a, b):
    """Manhattan distance."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    rows = len(grid)
    cols = len(grid[0])

    def in_bounds(node):
        x, y = node
        return 0 < x < cols and 0 < y < rows

    def passable(node):
        x, y = node
        return grid[y][x] == 0

    def neighbors(node):
        x, y = node
        candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
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

        for nb in neighbors(current):
            if nb in closed:
                continue
            tg = g_score[current] + 1
            if tg < g_score.get(nb, float('inf')):
                came_from[nb] = current
                g_score[nb] = tg
                heapq.heappush(open_heap, (tg + heuristic(nb, goal), counter, nb))
                counter += 1

    return None


def random_grid(rows, cols, obstacle_prob=0.25):
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
    """Normalized gradient of the distance function (points away from nearest obstacle)"""
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
    """Constraints enforced:
    radius(c) = dist(c, obstacles) - obstacle_radius  > 0
    ||c - path_point|| <= radius(c)   (original center stays inside the ball)
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

    # Candidate directions: away from nearest obstacle + a ring around it
    primary = obstacle_gradient(base, obstacle_centers)
    if np.linalg.norm(primary) < 1e-9:
        return best_c, best_r

    dirs = [primary]
    ang0 = np.arctan2(primary[1], primary[0])
    for k in range(1, n_dirs):
        a = ang0 + 2 * np.pi * k / n_dirs
        dirs.append(np.array([np.cos(a), np.sin(a)]))

    for g in dirs:
        # Exponential expansion to find a bracketing step
        lo = 0.0  # last known feasible step
        hi = None  # first known infeasible step
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

        # Bisection to refine the maximal feasible step
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

        # Update global best
        if last_r > best_r:
            best_c, best_r = last_feasible.copy(), last_r

    return best_c, best_r


def grow_free_balls(path, obstacle_centers, obstacle_radius=0.71):
    """The balls grow till they hit an obstacle on each side or their original center hits the radius"""
    # TODO: Could be optimized to grow bigger
    balls = []
    for p in path:
        c, r = max_free_ball_center(p, obstacle_centers, obstacle_radius)
        balls.append((c, r))
    return balls


def plot_grid_circles(grid, start, goal, path=None, show_free_balls=True):
    rows, cols = len(grid), len(grid[0])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Obstacles
    grey_radius = 0.5  # True obstacle radius
    red_radius = 0.71  # Safe distance from obstacle
    red_circles = []
    obstacle_centers = []
    for y, row in enumerate(grid):
        for x, col in enumerate(row):
            if col == 1:
                circle = plt.Circle((x, y), grey_radius, color='gray', zorder=1)
                ax.add_patch(circle)
                obstacle_centers.append((x, y))

                circle_poly = Point(x, y).buffer(red_radius, resolution=32)
                red_circles.append(circle_poly)

    wall_centers = []
    for x in range(cols + 1):
        wall_centers.append((float(x), 0.0))
        wall_centers.append((float(x), float(rows)))
    for y in range(1, rows):
        wall_centers.append((0.0, float(y)))
        wall_centers.append((float(cols), float(y)))

    for x, y in wall_centers:
        ax.add_patch(plt.Circle((x, y), grey_radius, color='gray', zorder=1))
        obstacle_centers.append((x, y))
        red_circles.append(Point(x, y).buffer(red_radius, resolution=32))

    if red_circles:
        unified_red_boundary = unary_union(red_circles)

        if unified_red_boundary.geom_type == 'Polygon':
            polygons = [unified_red_boundary]
        else:
            polygons = list(unified_red_boundary.geoms)

        for poly in polygons:
            ex_x, ex_y = poly.exterior.xy
            ax.plot(ex_x, ex_y, color='red', linewidth=2, zorder=0)
            for interior in poly.interiors:
                in_x, in_y = interior.xy
                ax.plot(in_x, in_y, color='red', linewidth=2, zorder=0)

    if path and show_free_balls:
        balls = grow_free_balls(path, obstacle_centers, obstacle_radius=red_radius)

        # Plot
        for p, (c, r) in zip(path, balls):
            if r > 0.01:
                fb = plt.Circle(c, r, fill=False, edgecolor=(0.592, 0.631, 0.043), linewidth=1.5, linestyle='--', zorder=2)
                ax.add_patch(fb)
                if np.linalg.norm(np.array(c) - np.array(p)) > 0.02:
                    ax.annotate('', xy=c, xytext=p, arrowprops=dict(arrowstyle='->', color='purple', lw=0.8), zorder=3)

    # Start and goal
    ax.plot(*start, 'o', ms=12, mfc=(0.145, 0.678, 0.8), mec='black', label='Start')
    ax.plot(*goal, 'o', ms=12, mfc=(0.8, 0.145, 0.671), mec='black', label='Goal')

    if path:
        xs = [p[0] for p in path]; ys = [p[1] for p in path]
        ax.plot(xs, ys, '-', color=(0.282, 0.322, 0.329), lw=3, label='A*')
        ax.plot(xs, ys, 'o', color=(0.184, 0.196, 0.2), ms=4)

    ax.legend(loc='lower right')
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


rows, cols = 10, 10
start = (1, 1)
goal = (9, 9)
grid = random_grid(rows, cols, obstacle_prob=0.2)

grid[start[1]][start[0]] = 0
grid[goal[1]][goal[0]] = 0

# Loop till random gives us something with an actual path
while True:
    path = a_star(grid, start, goal)
    if path:
        path = densify_path(path, spacing=0.33)  # For more circles along the path
        plot_grid_circles(grid, start, goal, path, show_free_balls=True)
        break
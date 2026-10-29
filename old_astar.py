import heapq
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
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
        return 0 <= x < cols and 0 <= y < rows

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
            tg = g_score[current] + 1  # Tentative g-score
            if tg < g_score.get(nb, float('inf')):
                came_from[nb] = current
                g_score[nb] = tg
                heapq.heappush(open_heap, (tg + heuristic(nb, goal), counter, nb))
                counter += 1

    return None


def random_grid(rows, cols, obstacle_prob=0.25):
    return [[1 if random.random() < obstacle_prob else 0
             for _ in range(cols)] for _ in range(rows)]


def plot_grid_circles(grid, start, goal, path=None):
    rows, cols = len(grid), len(grid[0])

    fig, ax = plt.subplots(figsize=(8, 8))

    grey_radius = 0.5
    red_radius = 0.71
    red_circles = []

    for y, row in enumerate(grid):
        for x, col in enumerate(row):
            if col == 1:
                circle = plt.Circle((x, y), grey_radius, color='gray', zorder=1)
                ax.add_patch(circle)
                
                circle_poly = Point(x, y).buffer(red_radius, resolution=32)
                red_circles.append(circle_poly)

    if red_circles:
        unified_red_boundary = unary_union(red_circles)
        
        if unified_red_boundary.geom_type == 'Polygon':
            polygons = [unified_red_boundary]
        else:
            polygons = list(unified_red_boundary.geoms)
            
        for poly in polygons:
            # Exterior outline
            ex_x, ex_y = poly.exterior.xy
            ax.plot(ex_x, ex_y, color='red', linewidth=2, zorder=0)
            
            # Plot any interior holes if they exist
            for interior in poly.interiors:
                in_x, in_y = interior.xy
                ax.plot(in_x, in_y, color='red', linewidth=2, zorder=0)

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.set_axisbelow(True)  # Put the grid behind
    ax.grid(which='minor', color='lightgray', linewidth=0.5, zorder=-1)
    ax.tick_params(which='minor', length=0)
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.tick_params(axis='both', which='major', labelsize=8)

    ax.plot(*start, 'o', ms=14, mfc='lime', mec='black', label='Start')
    ax.plot(*goal,  'o', ms=14, mfc='red',  mec='black', label='Goal')

    if path:
        xs = [p[0] for p in path]; ys = [p[1] for p in path]
        ax.plot(xs, ys, '-', color='dodgerblue', lw=3, label='Path')
        ax.plot(xs, ys, 'o', color='dodgerblue', ms=4)

    ax.legend(loc='lower right')
    ax.set_xlim(-0.5, cols - 0.5); ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

rows, cols = 10, 10
start = (0, 0)
goal  = (9, 9)
grid = random_grid(rows, cols, obstacle_prob=0.3)

# Making sure start and goal are free
grid[start[1]][start[0]] = 0
grid[goal[1]][goal[0]] = 0

path = a_star(grid, start, goal)
plot_grid_circles(grid, start, goal, path)
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
refpath = RefPath([
    Point(0.5, 0.3),
    Point(1.5, 0.8),
    Point(2.6, 1.8),
    Point(3.5, 3.2),
    Point(4.2, 4.8),
    Point(4.6, 6.0),
    Point(5.5, 6.8),
    Point(6.5, 7.3),
    Point(7.5, 7.8),
    Point(8.0, 8.0)])

circles = []
for p in refpath.points:
    circles.append(Circle(p.x, p.y, 1))


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


fig, ax = plt.subplots()

for c in circles:
    ax.add_patch(plt.Circle((c.x0, c.y0), radius=c.r, edgecolor=(0.592, 0.631, 0.043), fill=False, linewidth=3))

x_coords = [p.x for p in refpath.points]
y_coords = [p.y for p in refpath.points]
ax.plot(x_coords, y_coords, color='black', linewidth=2)

ax.set_aspect('equal', adjustable='datalim')
ax.autoscale_view()
ax.margins(0.05)
ax.autoscale_view()
plt.show()

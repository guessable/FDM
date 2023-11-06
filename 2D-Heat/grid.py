#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
#      "
#    '':''
#   ___:____      |\/|
# ,'        `.    \  /
# |  O        \___/  |
# ~^~^~^~^~^~^~^~^~^~^

import numpy as np
import matplotlib.pyplot as plt

from problem import Problem


class Grid(Problem):
    def __init__(self, domain: list, dx: float, dy: float, dt: float, a=1, b=1, case=1):
        super().__init__(domain, a, b, case)
        self.dx = dx
        self.dy = dy
        self.dt = dt

        self.nx = int((self.x_max - self.x_min) / dx)
        self.ny = int((self.y_max - self.y_min) / dy)
        self.nt = int((self.t1 - self.t0) / dt)

        self.x = np.linspace(self.x_min, self.x_max, self.nx)
        self.y = np.linspace(self.y_min, self.y_max, self.ny)
        self.t = np.linspace(self.t0, self.t1, self.nt)

    def u_bc_ic(self, u_num):
        T, Y = np.meshgrid(self.t, self.y)
        u_num[0, :, :] = self.bc_x0(Y, T)
        u_num[-1, :, :] = self.bc_x1(Y, T)

        T, X = np.meshgrid(self.t, self.x)
        u_num[:, 0, :] = self.bc_y0(X, T)
        u_num[:, -1, :] = self.bc_y1(X, T)

        Y, X = np.meshgrid(self.y, self.x)
        u_num[:, :, 0] = self.ic(X, Y)

        return u_num


if __name__ == "__main__":
    domain = [0, 2, 0, 2, 0, 1]
    grid = Grid(domain, 0.01, 0.01, 0.01)
    u_num = np.zeros((grid.nx, grid.ny, grid.nt))
    print(grid.u_bc_ic(u_num))

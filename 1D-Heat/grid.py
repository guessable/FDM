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
    def __init__(self, domain: list, dx: float, dt: float, a=1, case=1) -> None:
        super().__init__(domain, a, case)
        self.dx = dx
        self.dt = dt

        self.nx = int((self.x_max - self.x_min) / dx)
        self.nt = int((self.t_end - self.t_begin) / dt)

        self.x = np.linspace(self.x_min, self.x_max, self.nx)
        self.t = np.linspace(self.t_begin, self.t_end, self.nt)

        self.T, self.X = np.meshgrid(self.t, self.x)

    def bc_ic(self, u_num: np.ndarray) -> np.ndarray:
        u_num[:, 0] = self.IC(self.x)

        u_num[0, :] = self.bc0(self.t)
        u_num[self.nx - 1, :] = self.bc1(self.t)
        return u_num

    def plot_grid(self) -> None:
        ax = plt.subplot()
        ax.scatter(self.X[1:-1], self.T[1:-1], marker="*")
        ax.scatter(self.X[[0, -1], :], self.T[[0, -1], :], marker="^", label="BC")
        ax.scatter(self.X[:, 0], self.T[:, 0], label="IC")

        ax.set_xlabel(r"x")
        ax.set_ylabel(r"t")

        ax.legend()

        plt.show()


if __name__ == "__main__":
    domain = [0, 2, 0, 1]
    grid = Grid(domain, 0.05, 0.05)
    grid.plot_grid()

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


class Grid:
    def __init__(self, domain, dx, dt):
        self.x_min, self.x_max, self.t_begin, self.t_end = domain
        self.dx = dx
        self.dt = dt

        self.nx = int((self.x_max - self.x_min) / dx)
        self.nt = int((self.t_end - self.t_begin) / dt)

        self.x = np.linspace(self.x_min, self.x_max, self.nx)
        self.t = np.linspace(self.t_begin, self.t_end, self.nt)

        self.T, self.X = np.meshgrid(self.t, self.x)

    def show_grid(self):
        fig, ax = plt.subplots()
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
    grid.show_grid()

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
    def __init__(self, domain: list, h: float, case=1) -> None:
        super().__init__(case)
        self.h = h
        self.x_min, self.x_max, self.y_min, self.y_max = domain
        self.nx = int((self.x_max - self.x_min) / h)
        self.ny = int((self.y_max - self.y_min) / h)

        self.x = np.linspace(self.x_min, self.x_max, self.nx)
        self.y = np.linspace(self.y_min, self.y_max, self.ny)

        self.Y, self.X = np.meshgrid(self.y, self.x)

    def plot_grid(self) -> None:
        ax = plt.subplot()
        ax.scatter(self.X[1:-1], self.Y[1:-1], marker="*")
        ax.scatter(self.X[[0, -1], :], self.Y[[0, -1], :], marker="^", label="BC")
        ax.scatter(self.X[:, [0, -1]], self.Y[:, [0, -1]], label="BC")

        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")

        ax.legend()

        plt.show()


if __name__ == "__main__":
    grid = Grid([0, 1, 0, 1], 0.1)
    grid.plot_grid()

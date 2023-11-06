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


class Problem:
    """
    u_t + au_x + bu_y = 0
    """

    def __init__(self, domain: list, a=1, b=1, case=1):
        self.x_min, self.x_max, self.y_min, self.y_max, self.t0, self.t1 = domain
        self.case = case
        self.a = a
        self.b = b

    def ic(self, x, y):
        match self.case:
            case 1:
                ic = np.ones_like(x)
                ic[(x >= 0.5) & (y >= 0.5)] = 0
                return ic
            case _:
                return 0

    def solution(self, x, y, t):
        return self.ic(x - self.a * t, y - self.b * t)

    def bc_x0(self, y, t):
        return self.solution(self.x_min, y, t)

    def bc_x1(self, y, t):
        return self.solution(self.x_max, y, t)

    def bc_y0(self, x, t):
        return self.solution(x, self.y_min, t)

    def bc_y1(self, x, t):
        return self.solution(x, self.y_max, t)


if __name__ == "__main__":
    domain = [0, 1, 0, 1, 0, 1]
    x = np.linspace(0, 2, 50)
    y = np.linspace(0, 2, 50)
    t = np.linspace(0, 1, 50)

    X, Y = np.meshgrid(x, y)
    problem = Problem(domain)

    ic = problem.ic(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.contourf(X, Y, ic)
    plt.show()

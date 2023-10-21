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

from grid import Grid


class Problem(Grid):
    """
    u_t + a*u_x = 0
    """

    def __init__(self, domain, dx, dt, a=1, case=1) -> None:
        super().__init__(domain, dx, dt)
        self.a = a
        self.case = case

    def IC(self, x):
        if self.case == 1:
            ic = np.exp(-160 * (x - 1.5) ** 2)
            idx = np.where((x >= 0.3) & (x <= 0.5))
            ic[idx] = 1
            return ic

    def solution(self, x, t):
        return self.IC(x - self.a * t)

    def bc0(self, t):
        return self.solution(self.x_min, t)

    def bc1(self, t):
        return self.solution(self.x_max, t)


if __name__ == "__main__":
    domain = [0, 2, 0, 1]
    problem = Problem(domain, 0.01, 0.01, case=1)

    fig, ax = plt.subplots()
    X, T = problem.X, problem.T
    ax.contourf(X, T, problem.solution(X, T))
    ax.set_title(f"case:{problem.case}")
    plt.show()

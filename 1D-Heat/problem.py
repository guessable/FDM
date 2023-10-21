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
    u_t = a * u_xx + f(x,t)
    """

    def __init__(self, domain, dx, dt, a=1, case=1) -> None:
        super().__init__(domain, dx, dt)
        self.a = a
        self.case = case

    def solution(self, x, t):
        if self.case == 0:
            return 1
        elif self.case == 1:
            return x * (1 - x) - 2 * t
        elif self.case == 2:
            return x**2 + t**2
        elif self.case == 3:
            return 10 * np.sin(np.pi * x) * np.sin(np.pi * t)
        elif self.case == 4:
            return np.sin(np.pi * (x + t))

    def f(self, x, t):
        if self.case == 0:
            return 0
        elif self.case == 1:
            return 0
        elif self.case == 2:
            return 2 * t - 2
        elif self.case == 3:
            return (
                10
                * np.pi
                * np.sin(np.pi * x)
                * (np.cos(np.pi * t) + self.a * np.pi * np.sin(np.pi * t))
            )
        elif self.case == 4:
            return np.pi * (
                np.cos(np.pi * (x + t)) + self.a * np.pi * (np.sin(np.pi * (x + t)))
            )

    def bc0(self, t):
        return self.solution(self.x_min, t)

    def bc1(self, t):
        return self.solution(self.x_max, t)

    def IC(self, x):
        return self.solution(x, self.t_begin)


if __name__ == "__main__":
    domain = [0, 2, 0, 1]
    problem = Problem(domain, 0.01, 0.01, case=1)

    fig, ax = plt.subplots()
    X, T = problem.X, problem.T
    ax.contourf(X, T, problem.solution(X, T))
    ax.set_title(f"case:{problem.case}")
    plt.show()

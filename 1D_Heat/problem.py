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
    u_t = a * u_xx + f(x,t)
    """

    def __init__(self, domain, a=1, case=1) -> None:
        self.x_min, self.x_max, self.t_begin, self.t_end = domain
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
    x = np.linspace(0, 2, 100)
    t = np.linspace(0, 1, 100)

    X, T = np.meshgrid(x, t)

    problem = Problem([0, 2, 0, 1], case=4)

    fig, ax = plt.subplots()
    ax.contourf(X, T, problem.solution(X, T))
    plt.show()

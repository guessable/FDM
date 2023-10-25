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
    u_t + a*u_x = 0
    """

    def __init__(self, domain, a=1, case=1) -> None:
        self.x_min, self.x_max, self.t_begin, self.t_end = domain
        self.a = a
        self.case = case

    def IC(self, x):
        match self.case:
            case 1:
                ic = np.exp(-160 * (x - 1.5) ** 2)
                ic[np.where((x >= 0.25) & (x <= 0.75))] = 1
                return ic
            case 2:
                ic = np.zeros_like(x)
                ic[np.where(x <= 1)] = 1
                return ic
            case 3:
                ic = np.zeros_like(x)
                ic[np.where(x <= 1)] = 2
                ic[np.where((x > 1) & (x <= 2))] = 1
                return ic
            case _:
                return 0

    def solution(self, x, t):
        return self.IC(x - self.a * t)

    def bc0(self, t):
        return self.solution(self.x_min, t)

    def bc1(self, t):
        return self.solution(self.x_max, t)


if __name__ == "__main__":
    domain = [0, 2, 0, 1]
    problem = Problem(domain, case=1)

    x = np.linspace(0, 2, 100)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contourf(X, T, problem.solution(X, T))
    ax.set_title(f"case:{problem.case}")
    plt.show()

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

    def __init__(self, domain: list, a=1, case=1) -> None:
        self.x_min, self.x_max, self.t_begin, self.t_end = domain
        self.a = a
        self.case = case

    def solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray | int:
        match self.case:
            case 1:
                return x * (1 - x)
            case 2:
                return x**2 + t**2
            case 3:
                return 10 * np.sin(np.pi * x) * np.sin(np.pi * t)
            case 4:
                return np.sin(np.pi * (x + t))
            case 5:
                return np.sin(100 * (x - t))
            case _:
                return 0

    def f(self, x: np.ndarray, t: np.ndarray) -> np.ndarray | int:
        match self.case:
            case 1:
                return 2
            case 2:
                return 2 * t - 2
            case 3:
                return (
                    10
                    * np.pi
                    * np.sin(np.pi * x)
                    * (np.cos(np.pi * t) + self.a * np.pi * np.sin(np.pi * t))
                )
            case 4:
                return np.pi * (
                    np.cos(np.pi * (x + t)) + self.a * np.pi * (np.sin(np.pi * (x + t)))
                )
            case 5:
                return 10000 * np.sin(100 * (x - t)) - 100 * np.cos(100 * (x - t))
            case _:
                return 0

    def bc0(self, t: np.ndarray) -> np.ndarray:
        return self.solution(self.x_min, t)

    def bc1(self, t: np.ndarray) -> np.ndarray:
        return self.solution(self.x_max, t)

    def IC(self, x: np.ndarray) -> np.ndarray:
        return self.solution(x, self.t_begin)


if __name__ == "__main__":
    domain = [0, 2, 0, 1]
    problem = Problem(domain, case=1)

    x = np.linspace(0, 2, 100)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)

    fig = plt.figure()
    ax = fig.subplots()
    ax.contourf(X, T, problem.solution(X, T))
    ax.set_title(f"case:{problem.case}")
    plt.show()

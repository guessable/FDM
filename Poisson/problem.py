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
    -(u_xx + u_yy) = f
    """

    def __init__(self, case=1) -> None:
        self.case = case

    def solution(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | int:
        match self.case:
            case 0:
                return x + y + x * y
            case 1:
                return x**2 + y**2
            case 2:
                return np.exp(-(x**2 + y**2))
            case 3:
                return np.sin(np.pi * (x + y))
            case 4:
                return np.sqrt(x**2 + y**2)
            case 5:
                return np.sin(np.pi * (x**2 + y**2))
            case _:
                print("case not define")
                exit()

    def f(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | int:
        match self.case:
            case 0:
                return 0
            case 1:
                return -4
            case 2:
                return -(4 * x**2 + 4 * y**2 - 4) * np.exp(-(x**2 + y**2))
            case 3:
                return 2 * np.pi**2 * np.sin(np.pi * (x + y))
            case 4:
                return -1 / np.sqrt(x**2 + y**2)
            case 5:
                return -4 * np.pi * np.cos(
                    np.pi * (x**2 + y**2)
                ) + 4 * np.pi**2 * (x**2 + y**2) * np.sin(
                    np.pi * (x**2 + y**2)
                )
            case _:
                print("case not define")
                exit()


if __name__ == "__main__":
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    problem = Problem(case=1)
    solution = problem.solution(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot()
    con = ax.contourf(X, Y, solution)
    fig.colorbar(con)
    ax.set_title(f"{problem.case=}")

    plt.show()

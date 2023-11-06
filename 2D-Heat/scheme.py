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


class Scheme(Grid):
    def __init__(self, domain: list, dx: float, dy: float, dt: float, a=1, b=1, case=1):
        super().__init__(domain, dx, dy, dt, a, b, case)

        self.mu_x = self.dt / self.dx
        self.mu_y = self.dt / self.dy

    def solve(self, scheme: str):
        match scheme:
            case "UpWind":
                return self.__UpWind()

    def __UpWind(self):
        u_num = np.zeros((self.nx, self.ny, self.nt))
        u_num = self.u_bc_ic(u_num)

        for n in range(1, self.nt):
            for j in range(1, self.nx - 1):
                for k in range(1, self.ny - 1):
                    u_num[j, k, n] = (
                        u_num[j, k, n - 1]
                        - self.a
                        * self.mu_x
                        * (u_num[j, k, n - 1] - u_num[j - 1, k, n - 1])
                        - self.b
                        * self.mu_y
                        * (u_num[j, k, n - 1] - u_num[j, k - 1, n - 1])
                    )

        return u_num

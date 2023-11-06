#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
#      "
#    '':''
#   ___:____      |\/|
# ,'        `.    \  /
# |  O        \___/  |
# ~^~^~^~^~^~^~^~^~^~^

import numpy as np

from grid import Grid


class Scheme(Grid):
    def __init__(self, domain: list, dx: float, dt: float, a=1, case=1):
        super().__init__(domain, dx, dt, a, case)
        self.nu = dt / dx

    def solve(self, scheme: str) -> np.ndarray:
        match scheme:
            case "UpWind":
                return self.__UpWind()
            case "LaxWendroff":
                return self.__LaxWendroff()
            case "Wendroff":
                return self.__Wendroff()
            case "LeapFrog":
                return self.__LeapFrog()
            case "LaxFriedrichs":
                return self.__LaxFriedrichs()
            case "Carlson":
                return self.__Carlson()
            case _:
                print(f"{scheme} not define")
                exit()

    def __UpWind(self) -> np.ndarray:
        u_num = np.zeros((self.nx, self.nt))
        u_num = self.bc_ic(u_num)
        k = 0 if self.a > 0 else 1
        for n in range(1, self.nt):
            for i in range(1, self.nx - 1):
                u_num[i, n] = u_num[i, n - 1] - self.nu * self.a * (
                    u_num[i + k, n - 1] - u_num[i + k - 1, n - 1]
                )

        return u_num

    def __LaxWendroff(self) -> np.ndarray:
        u_num = np.zeros((self.nx, self.nt))
        u_num = self.bc_ic(u_num)
        for n in range(1, self.nt):
            for i in range(1, self.nx - 1):
                u_num[i, n] = (
                    u_num[i, n - 1]
                    - 0.5
                    * self.nu
                    * self.a
                    * (u_num[i + 1, n - 1] - u_num[i - 1, n - 1])
                    + 0.5
                    * self.nu**2
                    * self.a**2
                    * (u_num[i + 1, n - 1] - 2 * u_num[i, n - 1] + u_num[i - 1, n - 1])
                )

        return u_num

    def __LeapFrog(self) -> np.ndarray:
        u_num = np.zeros((self.nx, self.nt))
        u_num = self.bc_ic(u_num)
        k = 0 if self.a > 0 else 1
        for i in range(1, self.nx - 1):
            u_num[i, 1] = u_num[i, 0] - self.nu * self.a * (
                u_num[i + k, 0] - u_num[i + k - 1, 0]
            )

        for n in range(2, self.nt):
            for i in range(1, self.nx - 1):
                u_num[i, n] = u_num[i, n - 2] - self.a * self.nu * (
                    u_num[i + 1, n - 1] - u_num[i - 1, n - 1]
                )

        return u_num

    def __Wendroff(self) -> np.ndarray:
        u_num = np.zeros((self.nx, self.nt))
        u_num = self.bc_ic(u_num)
        for n in range(1, self.nt):
            for i in range(1, self.nx):
                u_num[i, n] = u_num[i - 1, n - 1] + (
                    (1 - self.a * self.nu) / (1 + self.a * self.nu)
                ) * (u_num[i, n - 1] - u_num[i - 1, n])

        return u_num

    def __LaxFriedrichs(self) -> np.ndarray:
        u_num = np.zeros((self.nx, self.nt))
        u_num = self.bc_ic(u_num)
        for n in range(1, self.nt):
            for i in range(1, self.nx - 1):
                u_num[i, n] = 0.5 * (
                    u_num[i - 1, n - 1] + u_num[i + 1, n - 1]
                ) - 0.5 * self.a * self.nu * (u_num[i + 1, n - 1] - u_num[i - 1, n - 1])
        return u_num

    def __Carlson(self) -> np.ndarray:
        u_num = np.zeros((self.nx, self.nt))
        u_num = self.bc_ic(u_num)
        for n in range(1, self.nt):
            for i in range(1, self.nx - 1):
                u_num[i, n] = (1 / (1 + self.a * self.nu)) * (
                    u_num[i, n - 1] + self.a * self.nu * u_num[i - 1, n]
                )
        return u_num


if __name__ == "__main__":
    solver = Scheme([0, 2, 0, 1], 0.01, 0.01)
    u_num = solver.solve("UpWind")

    X, T = solver.X, solver.T
    u_ref = solver.solution(X, T)

    print(f"max_error:{np.max(np.abs(u_ref-u_num)):.3e}")

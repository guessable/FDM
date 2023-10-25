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
    def __init__(self, scheme, domain, dx, dt, a=1, case=1):
        super().__init__(domain, dx, dt, a, case)
        self.nu = dt / dx
        self.scheme = scheme

    def solve(self):
        match self.scheme:
            case "UpWind":
                return self._UpWind()
            case "LaxWendroff":
                return self._LaxWendroff()
            case "Wendroff":
                return self._Wendroff()
            case "LeapFrog":
                return self._LeapFrog()
            case "LaxFriedrichs":
                return self._LaxFriedrichs()
            case _:
                print("scheme not define")
                exit()

    def _UpWind(self):
        k = 0 if self.a > 0 else 1
        for n in range(1, self.nt):
            for i in range(1, self.nx - 1):
                self.u_num[i, n] = self.u_num[i, n - 1] - self.nu * self.a * (
                    self.u_num[i + k, n - 1] - self.u_num[i + k - 1, n - 1]
                )

        return self.u_num

    def _LaxWendroff(self):
        for n in range(1, self.nt):
            for i in range(1, self.nx - 1):
                self.u_num[i, n] = (
                    self.u_num[i, n - 1]
                    - 0.5
                    * self.nu
                    * self.a
                    * (self.u_num[i + 1, n - 1] - self.u_num[i - 1, n - 1])
                    + 0.5
                    * self.nu**2
                    * self.a**2
                    * (
                        self.u_num[i + 1, n - 1]
                        - 2 * self.u_num[i, n - 1]
                        + self.u_num[i - 1, n - 1]
                    )
                )

        return self.u_num

    def _LeapFrog(self):
        k = 0 if self.a > 0 else 1
        for i in range(1, self.nx - 1):
            self.u_num[i, 1] = self.u_num[i, 0] - self.nu * self.a * (
                self.u_num[i + k, 0] - self.u_num[i + k - 1, 0]
            )

        for n in range(2, self.nt):
            for i in range(1, self.nx - 1):
                self.u_num[i, n] = self.u_num[i, n - 2] - self.a * self.nu * (
                    self.u_num[i + 1, n - 1] - self.u_num[i - 1, n - 1]
                )

        return self.u_num

    def _Wendroff(self):
        for n in range(1, self.nt):
            for i in range(1, self.nx):
                self.u_num[i, n] = self.u_num[i - 1, n - 1] + (
                    (1 - self.a * self.nu) / (1 + self.a * self.nu)
                ) * (self.u_num[i, n - 1] - self.u_num[i - 1, n])

        return self.u_num

    def _LaxFriedrichs(self):
        for n in range(1, self.nt):
            for i in range(1, self.nx - 1):
                self.u_num[i, n] = 0.5 * (
                    self.u_num[i - 1, n - 1] + self.u_num[i + 1, n - 1]
                ) - 0.5 * self.a * self.nu * (
                    self.u_num[i + 1, n - 1] - self.u_num[i - 1, n - 1]
                )
        return self.u_num


if __name__ == "__main__":
    solver = Scheme("UpWind", [0, 2, 0, 1], 0.01, 0.01)
    u_num = solver.solve()

    X, T = solver.X, solver.T
    u_ref = solver.solution(X, T)

    print(f"max_error:{np.max(np.abs(u_ref-u_num)):.3e}")

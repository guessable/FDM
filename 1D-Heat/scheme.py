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
    def __init__(self, scheme, domain, dx, dt, a=1, case=1) -> None:
        super().__init__(domain, dx, dt, a, case)
        self.mu = dt / dx**2
        self.scheme = scheme

    def solve(self):
        match self.scheme:
            case "explicit":
                return self._full_explicit()
            case "implicit":
                return self._full_implicit()
            case "Crank-Nicolson":
                return self._rose()
            case "Douglas":
                theta = 1 / 2 - (1 / (12 * self.mu * self.a))
                return self._rose(theta)
            case _:
                print("scheme is not define")
                exit()

    def _full_explicit(self):
        for n in range(1, self.nt):
            for i in range(1, self.nx - 1):
                xi, tn = self.x[i], self.t[n - 1]

                self.u_num[i, n] += (1 - 2 * self.a * self.mu) * self.u_num[i, n - 1]
                self.u_num[i, n] += self.dt * self.f(xi, tn)

                self.u_num[i, n] += self.a * self.mu * (self.u_num[i - 1, n - 1])
                self.u_num[i, n] += self.a * self.mu * (self.u_num[i + 1, n - 1])

        return self.u_num

    def _full_implicit(self):
        for n in range(1, self.nt):
            A = np.zeros((self.nx - 2, self.nx - 2))
            b = np.zeros(self.nx - 2)
            for i in range(1, self.nx - 1):
                xi, tn = self.x[i], self.t[n]
                A[i - 1, i - 1] += 1 + 2 * self.a * self.mu
                b[i - 1] += self.u_num[i, n - 1] + self.dt * self.f(xi, tn)

                if i == 1:
                    A[i - 1, i] += -self.a * self.mu
                    b[i - 1] += self.a * self.mu * self.bc0(self.t[n])
                elif i == self.nx - 2:
                    A[i - 1, i - 2] += -self.a * self.mu
                    b[i - 1] += self.a * self.mu * self.bc1(self.t[n])
                else:
                    A[i - 1, i] += -self.a * self.mu
                    A[i - 1, i - 2] += -self.a * self.mu

            u_n = np.linalg.solve(A, b)
            self.u_num[1 : self.nx - 1, n] = u_n

        return self.u_num

    def _rose(self, theta=0.5):
        for n in range(1, self.nt):
            A = np.zeros((self.nx - 2, self.nx - 2))
            b = np.zeros(self.nx - 2)
            for i in range(1, self.nx - 1):
                xi, tn = self.x[i], self.t[n]
                A[i - 1, i - 1] += 1 + 2 * theta * self.a * self.mu
                b[i - 1] += self.u_num[i, n - 1] + self.dt * self.f(xi, tn)
                b[i - 1] += (
                    (1 - theta)
                    * self.a
                    * self.mu
                    * (
                        self.u_num[i - 1, n - 1]
                        - 2 * self.u_num[i, n - 1]
                        + self.u_num[i + 1, n - 1]
                    )
                )

                if i == 1:
                    A[i - 1, i] += -theta * self.a * self.mu
                    b[i - 1] += theta * self.a * self.mu * self.bc0(self.t[n])
                elif i == self.nx - 2:
                    A[i - 1, i - 2] += -theta * self.a * self.mu
                    b[i - 1] += theta * self.a * self.mu * self.bc1(self.t[n])
                else:
                    A[i - 1, i] += -theta * self.a * self.mu
                    A[i - 1, i - 2] += -theta * self.a * self.mu

            u_n = np.linalg.solve(A, b)
            self.u_num[1 : self.nx - 1, n] = u_n

        return self.u_num


if __name__ == "__main__":
    domain = [0, 2, 0, 1]
    solver = Scheme("implicit", domain, 0.01, 0.01)

    X, T = solver.X, solver.T
    u_num = solver.solve()
    u_ref = solver.solution(X, T)

    print(f"max_error:{np.max(np.abs(u_ref-u_num)):.3e}")

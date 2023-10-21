#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
#      "
#    '':''
#   ___:____      |\/|
# ,'        `.    \  /
# |  O        \___/  |
# ~^~^~^~^~^~^~^~^~^~^

import argparse

import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation

from problem import Problem


class LW(Problem):
    def __init__(self, domain, dx, dt, a=1, case=1) -> None:
        super().__init__(domain, dx, dt, a, case)

        self.nu = dt / dx
        self.u_num = np.zeros((self.nx, self.nt))

    def solve(self):
        self.u_num[:, 0] = self.IC(self.x)

        self.u_num[0, :] = self.bc0(self.t)
        self.u_num[self.nx - 1, :] = self.bc1(self.t)

        return self.lw()

    def lw(self):
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


if __name__ == "__main__":
    # ==== args
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--a", type=int, default=1)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()

    domain = [0, 5, 0, 1]
    solver = LW(domain, args.dx, args.dt, args.a, args.case)

    X, T = solver.X, solver.T
    ua = solver.solve()
    sol = solver.solution(X, T)

    print(f"max_abs_error: {np.max(np.abs(ua-sol)):.3e}")

    # ==== plot
    fig, ax = plt.subplots(layout="constrained")
    (line1,) = ax.plot(solver.x, sol[:, 0], linestyle="-", label="Analytical")
    (line2,) = ax.plot(
        solver.x, ua[:, 0], linestyle="-.", label=rf"Lax-Wendroff ($\nu={solver.nu}$)"
    )
    ax.set_title("t=0")
    ax.legend()

    def update(frame):
        ymin = min(np.min(sol[:, frame]), np.min(ua[:, frame]))
        ymax = max(np.max(sol[:, frame]), np.max(ua[:, frame]))
        ax.set_ylim(ymin - 0.1, ymax + 0.1)
        line1.set_ydata(sol[:, frame])
        line2.set_ydata(ua[:, frame])
        ax.set_title(f"t={args.dt*frame:.3f}")

        return line1, line2

    ani = animation.FuncAnimation(
        fig, update, frames=solver.nt, interval=20, repeat=False
    )
    plt.show()

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
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from problem import Problem


class Classical(Problem):
    def __init__(self, domain, dx, dt, a=1, case=1) -> None:
        super().__init__(domain, a, case)

        self.dx = dx
        self.dt = dt
        self.mu = dt / dx**2
        self.nx = int((self.x_max - self.x_min) / dx)
        self.nt = int((self.t_end - self.t_begin) / dt)

        self.x = np.linspace(self.x_min, self.x_max, self.nx)
        self.t = np.linspace(self.t_begin, self.t_end, self.nt)

        self.tn, self.xn = np.meshgrid(self.t, self.x)
        self.u_num = np.zeros((self.nx, self.nt))

    def solve(self, type="implicit"):
        self.u_num[:, 0] = self.IC(self.x)

        self.u_num[0, :] = self.bc0(self.t)
        self.u_num[self.nx - 1, :] = self.bc1(self.t)

        if type == "implicit":
            print("====implicit====")
            return self.full_implicit()
        elif type == "explicit":
            print("====explicit====")
            return self.full_explicit()
        else:
            print("scheme_type=implicit or explicit")

    def full_explicit(self):
        for n in range(1, self.nt):
            for i in range(1, self.nx - 1):
                xi, tn = self.x[i], self.t[n - 1]

                self.u_num[i, n] += (1 - 2 * self.a * self.mu) * self.u_num[i, n - 1]
                self.u_num[i, n] += self.dt * self.f(xi, tn)

                self.u_num[i, n] += self.a * self.mu * (self.u_num[i - 1, n - 1])
                self.u_num[i, n] += self.a * self.mu * (self.u_num[i + 1, n - 1])

        return self.u_num

    def full_implicit(self):
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


if __name__ == "__main__":
    # ==== args
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=4)
    parser.add_argument("--a", type=int, default=1)
    parser.add_argument("--dx", type=float, default=0.001)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--scheme_type", type=str, default="implicit", help="implicit or explicit"
    )
    args = parser.parse_args()

    # ==== solver
    domain = [0, 2, 0, 1]
    solver = Classical(domain, args.dx, args.dt, a=args.a, case=args.case)

    X, T = solver.xn, solver.tn
    ua = solver.solve(type=args.scheme_type)
    sol = solver.solution(X, T)

    print(f"max_abs_error: {np.max(np.abs(ua-sol)):.3e}")

    # ==== plot
    fig, ax = plt.subplots(layout="constrained")
    (line1,) = ax.plot(solver.x, sol[:, 0], linestyle="-", label="Analytical")
    (line2,) = ax.plot(solver.x, ua[:, 0], linestyle="-.", label="Numerical")
    ax.set_title(f"case:{args.case}    t=0")
    ax.legend()

    def update(frame):
        ymin = min(np.min(sol[:, frame]), np.min(ua[:, frame]))
        ymax = max(np.max(sol[:, frame]), np.max(ua[:, frame]))
        ax.set_ylim(ymin - 0.1, ymax + 0.1)
        line1.set_ydata(sol[:, frame])
        line2.set_ydata(ua[:, frame])
        ax.set_title(f"case:{args.case}    t={args.dt*frame}")

        return line1, line2

    ani = animation.FuncAnimation(
        fig, update, frames=solver.nt, interval=20, repeat=False
    )
    plt.show()

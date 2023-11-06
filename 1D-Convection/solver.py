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

from scheme import Scheme


def compare(solver: Scheme, scheme: list) -> dict:
    X, T = solver.X, solver.T
    u_ref = solver.solution(X, T)

    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot()
    (line1,) = ax.plot(solver.x, u_ref[:, 0], linestyle="-", label="Reference")

    u_num = {sch: solver.solve(sch) for sch in scheme}
    lines = {
        sch: ax.plot(solver.x, u_num[sch][:, 10], linestyle="-.", label=f"{sch}")[0]
        for sch in scheme
    }

    ax.set_title(rf"$\nu$={solver.nu} t=0")
    ax.legend(loc="upper right")

    def update(frame):
        line1.set_ydata(u_ref[:, frame])
        updated_line = [
            line.set_ydata(u_num[sch][:, frame]) for sch, line in lines.items()
        ]
        ax.set_title(rf"$\nu$={solver.nu} t={solver.dt*frame:.3f}")

        return updated_line.append(line1)

    ani = animation.FuncAnimation(
        fig, update, frames=solver.nt, interval=20, repeat=False
    )

    # ani.save("arlson.gif")

    plt.show()

    return u_num


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--a", type=int, default=1)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.00625)
    parser.add_argument(
        "--scheme",
        nargs="*",
        default=["UpWind"],
        help="UpWind LaxWendroff LeapFrog Wendroff LaxFriedrichs Carlson",
    )
    args = parser.parse_args()

    domain = [0, 5, 0, 2]

    # solver
    scheme = args.scheme
    solver = Scheme(domain, args.dx, args.dt, args.a, args.case)

    # plot
    u_num = compare(solver, args.scheme)

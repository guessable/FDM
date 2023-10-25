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


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--a", type=int, default=1)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.00625)
    parser.add_argument(
        "--scheme",
        type=str,
        default="UpWind",
        help="--UpWind --LaxWendroff --LeapFrog --Wendroff --LaxFriedrichs",
    )
    args = parser.parse_args()

    # solver
    scheme = args.scheme
    domain = [0, 5, 0, 2]
    solver = Scheme(scheme, domain, args.dx, args.dt, args.a, args.case)

    # solve
    X, T = solver.X, solver.T
    u_num = solver.solve()
    u_ref = solver.solution(X, T)

    # error
    max_error = np.max(np.abs(u_ref - u_num))
    L2_error = np.linalg.norm(u_ref - u_num) / u_num.size

    print(f"scheme:{scheme}\n")
    print(f"{'max_error':<10} {'L2_error':<10}")
    print(f"{max_error:<10.3e} {L2_error:<10.3e}")
    # plot
    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot()
    (line1,) = ax.plot(solver.x, u_ref[:, 0], linestyle="-", label="Ref")
    (line2,) = ax.plot(
        solver.x,
        u_num[:, 0],
        linestyle="-.",
        label=rf"{scheme} ($\nu$={solver.nu:.3f})",
    )
    ax.set_title(f"case:{solver.case} t=0")
    ax.legend()

    def update(frame):
        ymin = min(np.min(u_ref[:, frame]), np.min(u_num[:, frame]))
        ymax = max(np.max(u_ref[:, frame]), np.max(u_num[:, frame]))
        ax.set_ylim(ymin - 0.1, ymax + 0.1)
        line1.set_ydata(u_ref[:, frame])
        line2.set_ydata(u_num[:, frame])
        ax.set_title(f"case:{solver.case} t={solver.dt*frame:.3f}")

        return line1, line2

    ani = animation.FuncAnimation(
        fig, update, frames=solver.nt, interval=20, repeat=False
    )
    plt.show()

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

from scheme import Scheme

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--show_non_zero", type=int, default=0, help="0 or 1")
    parser.add_argument(
        "--scheme", type=str, default="five", help="five nine oblique_five"
    )
    args = parser.parse_args()

    domain = [-1, 1, -1, 1]

    # solver
    solver = Scheme(domain, args.h, args.case, bool(args.show_non_zero))

    X, Y = solver.X, solver.Y
    u_num = solver.solve(args.scheme)
    u_ref = solver.solution(X, Y)

    # plot
    fig = plt.figure(figsize=(12, 5), layout="constrained")
    ax1 = fig.add_subplot(121)
    con1 = ax1.contourf(X, Y, u_num)
    fig.colorbar(con1)
    ax1.set_title(f"{args.scheme}")

    ax2 = fig.add_subplot(122)
    con2 = ax2.contourf(X, Y, u_ref)
    fig.colorbar(con2)
    ax2.set_title("ref")

    plt.show()

    max_error = np.max(np.abs(u_num - u_ref))
    print(f"max_error: {max_error:.3e}")

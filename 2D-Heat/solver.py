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

from scheme import Scheme

solver = Scheme([0, 1, 0, 1, 0, 1], 0.01, 0.01, 0.005)

u_num = solver.solve("UpWind")


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

Y, X = np.meshgrid(solver.y, solver.x)

co1 = ax1.contourf(X, Y, solver.solution(X, Y, 0.25))
co2 = ax2.contourf(X, Y, u_num[:, :, 50])
fig.colorbar(co1)
fig.colorbar(co2)

plt.show()


# print(u_num[:, :, 0].shape)

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


class Problem:
    """
    u_t = a*u_xx+b*u_yy + f(x,y,t)
    """

    def __init__(self, domain, t_begin, t_end, a=1, b=1, case=1):
        self.x_min, self.x_max, self.y_min, self.y_max = domain
        self.t_begin = t_begin
        self.t_end = t_end
        self.case = case
        self.a = a
        self.b = b

    def solution(self, x, y, t):
        match self.case:
            case 1:
                return np.sin(np.pi * (x + y + t))
            case _:
                return 0

    def f(self, x, y, t):
        match self.case:
            case 1:
                return np.pi * (
                    np.cos(np.pi * (x + y + t))
                    + self.a * np.pi * np.sin(np.pi * (x + y + t))
                    + self.b * np.pi * np.sin(np.pi * (x + y + t))
                )
            case _:
                return 0

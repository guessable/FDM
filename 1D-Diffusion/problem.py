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
    u_t = a(x,t) * u_xx
    or
    u_t = (a(x,t)*u_x)_x
    """

    def __init__(self, domain, case=1):
        self.x_min, self.x_max, self.t_begin, self.t_end = domain
        self.case = 1

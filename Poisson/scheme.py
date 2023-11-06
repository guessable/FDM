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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from grid import Grid


class Scheme(Grid):
    def __init__(self, domain: list, h: float, case=1, show_non_zero=False) -> None:
        super().__init__(domain, h, case)
        self.show_non_zero = show_non_zero

    def solve(self, scheme: str) -> np.ndarray:
        match scheme:
            case "five":
                u_num = self.five_point()
            case "oblique_five":
                u_num = self.oblique_five()
            case "nine":
                u_num = self.nine()
            case _:
                print("scheme not define")
                exit()
        return u_num.reshape(self.nx, self.ny)

    def idx(self, i: int, j: int) -> int:
        return i * self.ny + j

    def five_point(self) -> np.ndarray:
        indptr, indices, data = [0], [], []
        b = np.zeros(self.nx * self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                c = self.idx(i, j)
                if i in (0, self.nx - 1) or j in (0, self.ny - 1):
                    indptr.append(indptr[-1] + 1)
                    indices.append(c)
                    data.append(1)
                    b[c] += self.solution(self.x[i], self.y[j])
                else:
                    n, s, w, e = (
                        self.idx(i - 1, j),
                        self.idx(i + 1, j),
                        self.idx(i, j - 1),
                        self.idx(i, j + 1),
                    )
                    indptr.append(indptr[-1] + 5)
                    indices += [c, n, s, w, e]
                    data += [4 / self.h**2]
                    data += [-1 / self.h**2] * 4
                    b[c] += self.f(self.x[i], self.y[j])
        A = csr_matrix((data, indices, indptr))

        if self.show_non_zero:
            plt.matshow(A.toarray())

        u_num = spsolve(A, b)
        return u_num

    def oblique_five(self) -> np.ndarray:
        indptr, indices, data = [0], [], []
        b = np.zeros(self.nx * self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                c = self.idx(i, j)
                if i in (0, self.nx - 1) or j in (0, self.ny - 1):
                    indptr.append(indptr[-1] + 1)
                    indices.append(c)
                    data.append(1)
                    b[c] += self.solution(self.x[i], self.y[j])
                else:
                    ne, se, nw, sw = (
                        self.idx(i - 1, j + 1),
                        self.idx(i + 1, j + 1),
                        self.idx(i - 1, j - 1),
                        self.idx(i + 1, j - 1),
                    )
                    indptr.append(indptr[-1] + 5)
                    indices += [c, ne, se, nw, sw]
                    data += [4 / (2 * self.h**2)]
                    data += [-1 / (2 * self.h**2)] * 4
                    b[c] += self.f(self.x[i], self.y[j])
        A = csr_matrix((data, indices, indptr))

        if self.show_non_zero:
            plt.matshow(A.toarray())

        u_num = spsolve(A, b)
        return u_num

    def nine(self) -> np.ndarray:
        indptr, indices, data = [0], [], []
        b = np.zeros(self.nx * self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                c = self.idx(i, j)
                if i in (0, self.nx - 1) or j in (0, self.ny - 1):
                    indptr.append(indptr[-1] + 1)
                    indices.append(c)
                    data.append(1)
                    b[c] += self.solution(self.x[i], self.y[j])
                else:
                    n, s, w, e = (
                        self.idx(i - 1, j),
                        self.idx(i + 1, j),
                        self.idx(i, j - 1),
                        self.idx(i, j + 1),
                    )
                    ne, se, nw, sw = (
                        self.idx(i - 1, j + 1),
                        self.idx(i + 1, j + 1),
                        self.idx(i - 1, j - 1),
                        self.idx(i + 1, j - 1),
                    )
                    indptr.append(indptr[-1] + 9)
                    indices += [c, n, s, w, e, ne, se, nw, sw]
                    data += [10 / (3 * self.h**2)]
                    data += [-2 / (3 * self.h**2)] * 4
                    data += [-1 / (6 * self.h**2)] * 4

                    b[c] += (1 - self.h**2 / 3) * self.f(self.x[i], self.y[j])
                    b[c] += (
                        self.h**2
                        / 12
                        * (
                            self.f(self.x[i - 1], self.y[j])
                            + self.f(self.x[i + 1], self.y[j])
                            + self.f(self.x[i], self.y[j - 1])
                            + self.f(self.x[i], self.y[j + 1])
                        )
                    )
        A = csr_matrix((data, indices, indptr))

        if self.show_non_zero:
            plt.matshow(A.toarray())

        u_num = spsolve(A, b)
        return u_num


if __name__ == "__main__":
    scheme = Scheme([-1, 1, -1, 1], 0.05)
    u_num = scheme.solve("five")

    u_ref = scheme.solution(scheme.X, scheme.Y)

    fig = plt.figure()
    ax = fig.add_subplot()
    con = ax.contourf(scheme.X, scheme.Y, np.abs(u_num - u_ref))
    fig.colorbar(con)
    ax.set_title("max_error")
    plt.show()

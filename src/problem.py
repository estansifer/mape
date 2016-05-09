import numpy as np
import scipy.interpolate as sint

import thermo

def array1d(x):
    x_ = np.array(x, copy = True)
    if len(x_.shape) > 1:
        raise ValueError()
    if x_.shape == ():
        x_ = x_.reshape((1,))
    return x_

# def split(x, n):
    # x_ = []
    # for i in x:
        # x_ += [i] * n
    # return x_

class Problem:
    def __init__(self, y, p, s):
        self.y = array1d(y)
        self.p = array1d(p)
        self.s = array1d(s)
        self.n = len(self.y)
        if (self.n != len(self.p)) or (self.n != len(self.s)):
            raise ValueError()

        for k in range(self.n - 1):
            if self.p[k] > self.p[k + 1]:
                raise ValueError()

    def from_yps(y, p, s):
        return Problem(y, p, s)

    def from_ypT(y, p, T):
        y = array1d(y)
        p = array1d(p)
        T = array1d(T)

        n = len(y)
        s = np.zeros((n,))
        for i in range(n):
            s[i] = thermo.entropy(y[i], p[i], T[i])
        return Problem.from_yps(y, p, s)

    def h(self, j, k):
        return thermo.enthalpy(self.y[j], self.p[k], self.s[j])

    def h_dp(self, j, k):
        return thermo.enthalpy_dp(self.y[j], self.p[k], self.s[j])

    def h_d1p(self, j, k):
        if k + 1 == self.n:
            raise ValueError()
        else:
            return (self.h(j, k + 1) - self.h(j, k)) / (self.p[k + 1] - self.p[k])

    # Results in duplicate pressures, which messes up Lorenz
    # n -- number of times to duplicate each parcel (must be integer)
    # def splitA(self, n):
        # return Problem.from_yps(split(self.y, n), split(self.p, n), split(self.s, n))

    # n -- number of times to duplicate each parcel (must be integer)
    # def splitB(self, n):
        # dp = float(self.p[1] - self.p[0])
        # plow = self.p[0] - dp / 2
        # phigh = plow + dp * self.n
#
        # p = np.linspace(plow, phigh, self.n * n + 2)[1:-1]
        # return Problem.from_yps(split(self.y, n), p, split(self.s, n))

    # This is the preferred interpolation method, as it interpolates all of y, p, and s.
    # Interpolation is linear in y-p-s space (equivalently, also in w-p-s space).
    # n -- number of parcels to end up with
    # def splitC(self, n):
    def lin_interpolate(self, n):
        pnew = np.linspace(self.p[0], self.p[-1], n)
        ynew = sint.interp1d(self.p, self.y)(pnew)
        snew = sint.interp1d(self.p, self.s)(pnew)
        return Problem.from_yps(ynew, pnew, snew)

# 'j' refers to the index of the parcel, i.e., the index for the y and s arrays
# 'k' refers to the index of the pressure level, i.e. the index for the p array
class Solution:
    def __init__(self, p):
        self.problem = p
        self.n = p.n
        self.j2k = None
        self.k2j = None
        self.total_h = None

    def from_j2k(p, j2k):
        self = Solution(p)
        self.j2k = array1d(j2k)
        self.k2j = np.full((self.n,), -1, dtype = int)
        for j, k in enumerate(self.j2k):
            self.k2j[k] = j
        self.check_valid()
        return self

    def from_k2j(p, k2j):
        self = Solution(p)
        self.k2j = array1d(k2j)
        self.j2k = np.full((self.n,), -1, dtype = int)
        for k, j in enumerate(self.k2j):
            self.j2k[j] = k
        self.check_valid()
        return self

    def check_valid(self):
        if (self.j2k is None) or (self.k2j is None):
            raise ValueError()
        if (self.j2k.shape != (self.n,)) or (self.k2j.shape != (self.n,)):
            raise ValueError()
        if (self.j2k.dtype != int) or (self.k2j.dtype != int):
            raise ValueError()
        for i in range(self.n):
            k = self.j2k[i]
            j = self.k2j[i]
            if (k < 0) or (k >= self.n) or (j < 0) or (j >= self.n):
                raise ValueError()
            if (self.k2j[k] != i) or (self.j2k[j] != i):
                raise ValueError()

    def evaluate(self):
        if self.total_h is None:
            self.check_valid()
            total_h = 0
            for j in range(self.n):
                total_h += self.problem.h(j, self.j2k[j])
            self.total_h = total_h
        return self.total_h

    def similarity(self, other):
        return np.count_nonzero(self.j2k == other.j2k)

    def distance(self, other):
        p = self.problem
        d = 0
        for k in range(self.n):
            d += (p.h(self.k2j[k], k) - p.h(other.k2j[k], k)) ** 2
        return d

    def copy(self):
        return Solution.from_k2j(self.problem, self.k2j)

    # Returns a new Solution object with the parcels at pressure levels k1 and k2 swapped.
    # This is used for the Randall & Wang algorithm, which successively modifies a
    # configuration.
    def swap(self, k1, k2):
        k2j = np.array(self.k2j)
        t = k2j[k1]
        k2j[k1] = k2j[k2]
        k2j[k2] = t
        return Solution.from_k2j(self.problem, k2j)

    # Raise the parcel located at k1 to k2, k2 < k1, shifting down the parcels in between.
    # This is used for the Randall & Wang algorithm, which successively modifies a
    # configuration.
    def raise_parcel(self, k1, k2):
        if k1 == k2:
            return self
        if k1 < k2:
            raise ValueError()

        k2j = np.array(self.k2j)
        k2j[k2 + 1: k1 + 1] = self.k2j[k2 : k1]
        k2j[k2] = self.k2j[k1]
        return Solution.from_k2j(self.problem, k2j)

    def __str__(self):
        return 'j2k == {}\nk2j == {}'.format(self.j2k, self.k2j)

    def __eq__(self, other):
        return ((type(self) is type(other)) and
                np.array_equal(self.j2k, other.j2k) and np.array_equal(self.k2j, other.k2j))

    def __ne__(self, other):
        return not (self == other)

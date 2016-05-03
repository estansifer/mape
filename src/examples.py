import numpy as np
import timeit
import math

import thermo
import problem
import solvers

gettime = timeit.default_timer

def read_csv(filename):
    with open(filename, 'r') as f:
        xs = [float(x) for x in f.read().split(',')]
    return np.array(xs)

def read_problem(fileformat):
    f_p = fileformat.format('pressure')
    f_h = fileformat.format('relativehumidity')
    f_T = fileformat.format('temperature')

    p = read_csv(f_p)
    h = read_csv(f_h)
    T = read_csv(f_T)

    n = len(p)
    y = np.zeros(n)
    for i in range(n):
        p_star = thermo.compute_p_star(T[i])
        p_dry = p[i] - p_star * min(1, h[i])
        y[i] = h[i] * p_star / p_dry

    return problem.Problem.from_ypT(y, p, T)

def uniform(n, y_low = 0.001, y_high = 0.1, T_low = 250, T_high = 350, p_low = 1e4, p_high = 1e5):
    a = int(math.sqrt(n))
    b = ((n - 1) // a) + 1

    p = np.linspace(p_low, p_high, n)

    s_low = thermo.entropy(0, thermo.bar, T_low)
    s_high = thermo.entropy(0, thermo.bar, T_high)
    y_ = np.linspace(y_low, y_high, a)
    s_ = np.linspace(s_low, s_high, b)
    y, s = np.meshgrid(y_, s_, indexing = 'ij')

    return problem.Problem.from_yps(y.flatten()[:n], p, s.flatten()[:n])

# Result has 10 * n parcels
def pathological_problem(n, set_initial = False):
    y0 = 0.5
    y1 = 0
    # T0 = 335        # temperature at 1 atm
    # T1 = 375        # temperature at 1 atm
    # s0 = thermo.entropy(y0, thermo.atm, T0)
    # s1 = thermo.entropy(y1, thermo.atm, T1)
    T0 = 335        # temperature at 1 bar
    T1 = 375        # temperature at 1 bar
    s0 = thermo.entropy(y0, thermo.bar, T0)
    s1 = thermo.entropy(y1, thermo.bar, T1)

    y = np.array(([y0] * n) + ([y1] * (9 * n)))
    s = np.array(([s0] * n) + ([s1] * (9 * n)))
    p = np.linspace(4e4, 6.5e4, 10 * n)

    pr = problem.Problem.from_yps(y, p, s)

    if set_initial:
        # We re-order the parcels so that it starts in the worst initial condition.
        pr = reorder_worst(pr)

    return pr

def reorder_problem(p, solution):
    y = p.y[solution.k2j]
    s = p.s[solution.k2j]
    return problem.Problem.from_yps(y, p.p, s)

def reorder_worst(p):
    return reorder_problem(p, solvers.munkres(p, worst = True))

def reorder_best(p):
    return reorder_problem(p, solvers.munkres(p, worst = False))

def reorder_random(p):
    return reorder_problem(p, solvers.shuffle(p))

problem1 = read_problem('../data/01_{}')
problem2 = read_problem('../data/02_{}')

# problem2_randall_sol = problem.Solution.from_j2k(problem2, [0, 1] + list(range(3, 37)) + [2])

class Result:
    def __init__(self, p, solver):
        self.problem = p
        self.solver = solver
        self.name = None
        self.solution = None
        self.enthalpy = None
        self.score = None
        self.time = None

    def run(self):
        start = gettime()
        self.solution = self.solver(self.problem)
        end = gettime()
        self.time = end - start
        self.enthalpy = self.solution.evaluate()

    def __str__(self):
        return '{: <20}  {:<15.6g}  {:.2f}'.format(self.name, self.score, self.time)


class Results:
    def __init__(self, p, s = None):
        if s is None:
            s = dict(solvers.solvers)
        self.problem = p
        self.solvers = s
        self.results = None
        self.enthalpy_range = None          # J / kg

    def run(self):
        self.results = dict()
        min_h = None
        max_h = None
        for name in self.solvers:
            result = Result(self.problem, self.solvers[name])
            self.results[name] = result
            result.name = name
            print (name)
            result.run()
            if (min_h is None) or (min_h > result.enthalpy):
                min_h = result.enthalpy
            if (max_h is None) or (max_h < result.enthalpy):
                max_h = result.enthalpy

        for name in self.results:
            result = self.results[name]
            result.score = (result.enthalpy - min_h) / (max_h - min_h)
        # Divide by n to convert from enthalpy-per-kg-parcel to
        # enthalpy-per-kg-of-column-of-air, since there are n parcels in the column of air.
        # Units: J / kg
        self.enthalpy_range = (max_h - min_h) / self.problem.n

    def __str__(self):
        names = list(self.results.keys())
        names.sort(key = (lambda name : self.results[name].score))

        res = 'Problem size = {}, Enthalpy range = {:.6g} J / kg\n'.format(
                self.problem.n, self.enthalpy_range)
        res += (' ' * 22) + 'score            computation time (seconds)'
        for name in names:
            res += '\n'
            res += str(self.results[name])
        return res

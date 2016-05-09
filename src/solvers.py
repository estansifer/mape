import numpy as np
import math
import random

import problem

def initial(p):
    return problem.Solution.from_j2k(p, np.arange(p.n))

def shuffle(p):
    j2k = list(range(p.n))
    random.shuffle(j2k)
    return problem.Solution.from_j2k(p, j2k)

def shuffle_best(p, k = 1000):
    best = None
    for i in range(k):
        cur = shuffle(p)
        if best is None or best.evaluate() > cur.evaluate():
            best = cur
    return best

def greedy(p, from_bottom = False):

    js = set(range(p.n))
    k2j = np.full((p.n,), -1, dtype = int)

    ks = list(range(p.n))
    if from_bottom:
        ks.reverse()

    for k in ks:
        optimal_j = None

        if from_bottom:
            optimal_value = 1e300
            for j in js:
                temp = p.h_dp(j, k)
                if temp < optimal_value:
                    optimal_j = j
                    optimal_value = temp

        else:
            optimal_value = -1e300
            for j in js:
                temp = p.h_dp(j, k)
                if temp > optimal_value:
                    optimal_j = j
                    optimal_value = temp

        js.remove(optimal_j)
        k2j[k] = optimal_j

    return problem.Solution.from_k2j(p, k2j)

def divide_and_conquer(p):
    k2j = np.full((p.n,), -1, dtype = int)

    # Given a range [k_low, k_high) of pressure levels and a list js
    # of parcels to be assigned to those levels, perform the assignment.
    def subsolve(js, k_low, k_high):
        if len(js) != (k_high - k_low):
            raise ValueError(len(js), k_low, k_high)
        if k_low + 1 == k_high:
            k2j[k_low] = js[0]
            return

        k_mid = (k_low + k_high) // 2

        js.sort(key = (lambda j : p.h_dp(j, k_mid)), reverse = True)

        subsolve(js[:(k_mid - k_low)], k_low, k_mid)
        subsolve(js[(k_mid - k_low):], k_mid, k_high)

    subsolve(list(range(p.n)), 0, p.n)
    return problem.Solution.from_k2j(p, k2j)

def lorenz(p):
    js1 = list(range(p.n))
    js2 = list(range(p.n))
    # Sorting by dh/dp at a fixed pressure level is same as sorting by virtual temperature,
    # since the two are related by a constant.
    js1.sort(key = (lambda j : p.h_dp(j, 0)), reverse = True)
    js2.sort(key = (lambda j : p.h_dp(j, p.n - 1)), reverse = True)

    js_remaining = set(range(p.n))

    k2j = np.full((p.n,), -1, dtype = int)

    ji1 = 0
    ji2 = 0

    for k in range(p.n):
        while not (js1[ji1] in js_remaining):
            ji1 += 1
        while not (js2[ji2] in js_remaining):
            ji2 += 1
        j1 = js1[ji1]
        j2 = js2[ji2]
        if (j1 != j2) and p.h_d1p(j2, k) > p.h_d1p(j1, k):
            j1 = j2
        js_remaining.remove(j1)
        k2j[k] = j1

    return problem.Solution.from_k2j(p, k2j)

def randallwang(p):
    cache_h_dp = {}
    def h_dp(j, k):
        if (j, k) not in cache_h_dp:
            cache_h_dp[(j, k)] = p.h_dp(j, k)
        return cache_h_dp[(j, k)]

    cache_h = {}
    def h(j, k):
        if (j, k) not in cache_h:
            cache_h[(j, k)] = p.h(j, k)
        return cache_h[(j, k)]

    # based on Solution.evaluate(self)
    def cached_evaluate(config):
        config.check_valid()
        total_h = 0
        for j in range(config.n):
            total_h += h(j, config.j2k[j])
        return total_h

    config = initial(p)

    for k in range(p.n - 1):
        max_h_dp_a = h_dp(config.k2j[k], k)
        max_h_dp_b = h_dp(config.k2j[k], p.n - 1)
        k_a = k
        k_b = k

        for k2 in range(k + 1, p.n):
            h_dp_a = h_dp(config.k2j[k2], k)
            h_dp_b = h_dp(config.k2j[k2], p.n - 1)

            if h_dp_a > max_h_dp_a:
                max_h_dp_a = h_dp_a
                k_a = k2
            if h_dp_b > max_h_dp_b:
                max_h_dp_b = h_dp_b
                k_b = k2

        config_a = config.raise_parcel(k_a, k)
        config_b = config.raise_parcel(k_b, k)

        # if config_a.evaluate() < config_b.evaluate():
        if cached_evaluate(config_a) < cached_evaluate(config_b):
            config = config_a
        else:
            config = config_b

    return config

def randallwangsolution(p):
    assert (p.n == 37)
    return problem.Solution.from_k2j(p, [0, 1, 36] + list(range(2, 36)))

def munkres(p, worst = False):
    n = p.n
    cost = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            cost[j, k] = p.h(j, k)

    if worst:
        cost *= -1

    # Reduce cost by row and column
    cost -= np.amin(cost, axis = 1, keepdims = True)
    cost -= np.amin(cost, axis = 0, keepdims = True)

    infinity = max(1e10, (np.abs(np.max(cost)) + 1) * 1e10)

    j2k = np.full((n,), -1, dtype = int)
    k2j = np.full((n,), -1, dtype = int)

    j_potential = np.zeros((n,))
    k_potential = np.zeros((n,))

    for j in range(n):
        dist = cost[j, :] + k_potential[:] - j_potential[j]
        j_visited = np.zeros((n,), dtype = bool) # fills with False
        k_visited = np.zeros((n,), dtype = bool)
        j_visited[j] = True
        k2nearest_j = np.full((n,), j, dtype = int)

        while True:
            k = np.argmin(dist)
            slack = dist[k]
            j_potential[j_visited] += slack
            k_potential[k_visited] += slack
            dist[~k_visited] -= slack

            if k2j[k] < 0:
                while k >= 0:
                    j_ = k2nearest_j[k]
                    k_ = j2k[j_]
                    j2k[j_] = k
                    k2j[k] = j_
                    k = k_
                break
            else:
                j_ = k2j[k]
                j_visited[j_] = True
                k_visited[k] = True
                dist[k] = infinity

                dist_ = cost[j_, :] + k_potential[:] - j_potential[j_]
                ks_ = (~k_visited) & (dist_ < dist)
                dist[ks_] = dist_[ks_]
                k2nearest_j[ks_] = j_

    return problem.Solution.from_k2j(p, k2j)

# A solver is a function that takes a Problem and returns a Solution

solvers = {
    'initial' : initial,
    # 'random' : shuffle,
    # 'random-100' : (lambda p : shuffle_best(p, 100)),
    'greedy-from-top' : (lambda p : greedy(p, False)),
    # 'greedy-from-bottom' : (lambda p : greedy(p, True)),
    'divide-and-conquer' : divide_and_conquer,
    'lorenz' : lorenz,
    'randallwang' : randallwang,
    # 'randallwangsolution' : randallwangsolution,
    'munkres' : (lambda p : munkres(p, False)),
    'munkres-worst' : (lambda p : munkres(p, True))
        }

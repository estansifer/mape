import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import thermo
import problem
import solvers
import examples


def make_figure2():
    plt.rc('font', size = 17)
    plt.rc('text', usetex = True)
    # Values for the pathological problem
    y0 = 0.5
    y1 = 0
    T0 = 335        # temperature at 1 bar
    T1 = 375        # temperature at 1 bar
    s0 = thermo.entropy(y0, thermo.bar, T0)
    s1 = thermo.entropy(y1, thermo.bar, T1)

    print ('value of w for wet parcel', thermo.compute_w(y0))

    n = 10000
    p = np.linspace(4e4, 6.5e4, n)
    dhdp0 = np.zeros((n,))
    dhdp1 = np.zeros((n,))
    Tv0 = np.zeros((n,))
    Tv1 = np.zeros((n,))

    for i in range(n):
        dhdp0[i] = thermo.enthalpy_dp(y0, p[i], s0)
        dhdp1[i] = thermo.enthalpy_dp(y1, p[i], s1)
        Tv0[i] = thermo.virtual_temperature(y0, thermo.bar, s0)
        Tv1[i] = thermo.virtual_temperature(y1, thermo.bar, s1)

    plt.clf()
    plt.plot(dhdp1, p / 100, '-', label = 'dry parcel')
    plt.plot(dhdp0, p / 100, '--', label = 'wet parcel')
    plt.xlabel("{\\huge $\\partial_p h$} (J kg$^{-1}$ Pa$^{-1}$)")
    plt.ylabel("Pressure (hPa)")
    plt.ylim(670, 380)
    plt.legend(loc = 'lower right', frameon = False)

    ax = plt.gca()
    ax.tick_params(direction='out', right=False, top = False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplots_adjust(bottom = 0.15, left = 0.125)
    plt.savefig('../output/Stansifer_Fig2.png')
    plt.close()

    # plt.clf()
    # plt.plot(p / 100, Tv1, '-', label = 'dry air')
    # plt.plot(p / 100, Tv0, '--', label = 'wet air')
    # plt.xlabel("Pressure (hPa)")
    # plt.ylabel("virtual potential temperature (K)")
    # plt.xlim(380, 670)
    # plt.legend()
    # plt.savefig('../output/Stansifer_Fig_Other3.png')
    # plt.close()

def make_figure1():
    plt.rc('font', size = 13)
    plt.rc('text', usetex = True)
    p1 = examples.problem1
    # sol = solvers.divide_and_conquer(p1)
    sol = solvers.munkres(p1)

    y = np.array(p1.y)
    s = np.array(p1.s)
    p = np.zeros((len(y),))
    pt = np.zeros((len(y),))

    for i in range(len(y)):
        p[i] = p1.p[sol.j2k[i]]
        pt[i] = thermo.temperature(y[i], thermo.bar, s[i])

    # cmap = plt.get_cmap('hot')
    # cmap = plt.get_cmap('gray')

    # plt.clf()
    # sc = plt.scatter(pt, y, c = p / 100, cmap = cmap)
    # plt.colorbar(sc)
    # plt.xlabel("Potential temperature (K at 1000 hPa)")
    # plt.ylabel("Water content (molar fraction)")
    # plt.title("Pressure (in mbar) distribution at minimum enthalpy")
    # plt.xlim(270, 500)
    # plt.ylim(0, 0.025)
    # plt.savefig('../output/Stansifer_Fig_Other.png')
    # plt.close()

    # plt.clf()
    # sc = plt.scatter(pt, y, c = p / 100, cmap = cmap)
    # plt.colorbar(sc)
    # plt.xlabel("Potential temperature (K at 1000 hPa)")
    # plt.ylabel("Water content (molar fraction)")
    # plt.title("Pressure (in mbar) distribution at minimum enthalpy")
    # plt.xlim(270, 350)
    # plt.ylim(0, 0.025)
    # plt.savefig('../output/Stansifer_Fig_Other2.png')
    # plt.close()

    pt_ = pt.reshape(40, 40)
    y_ = y.reshape(40, 40)
    p_ = p.reshape(40, 40)

    p_ = np.array(p_, copy = True)
    p_[pt_ > 330] = np.nan

    plt.clf()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(pt_, y_ * thermo.epsilon, p_ / 100, rstride = 1, cstride = 1)
    ax.set_xlim(275, 321)
    ax.set_ylim(0.003 * thermo.epsilon, 0.022 * thermo.epsilon)
    ax.set_zlim(1000, 200)
    ax.view_init(elev=30, azim=160)
    ax.set_xlabel("{\\tiny .} \nTemperature (K) at 1000 hPa")
    ax.set_ylabel("{\\tiny .} \nWater content (kg per kg)")
    ax.set_zlabel("Pressure (hPa)")
    plt.subplots_adjust(bottom = 0.1, right = 1, left = 0)
    plt.savefig('../output/Stansifer_Fig1.png')
    plt.close()

def compute(problems = None):
    if problems is None:
        problems = [examples.problem2.lin_interpolate(1000),
                examples.problem1,
                examples.pathological_problem(100, True)]

    runs = [examples.Results(p) for p in problems]
    for r in runs:
        print ('Running all solvers on a problem...', end = '', flush = True)
        r.run()
        print (' done.')

    # print ('Initial')
    # print (runs[0].results['initial'].solution)
    # print ('R&W Hard-coded')
    # print (runs[0].results['randallwangsolution'].solution)
    # print ('R&W')
    # print (runs[0].results['randallwang'].solution)
    # print ('Lorenz')
    # print (runs[0].results['lorenz'].solution)
    # print ('Munkres')
    # print (runs[0].results['munkres'].solution)
    # print ('Divide and conquer')
    # print (runs[0].results['divide-and-conquer'].solution)

    return runs

def print_results(runs, prec = 5):
    import sys
    import numpy.version
    import scipy.version

    print ("Python version ", sys.version)
    print ("Numpy version ", numpy.version.full_version)
    print ("Scipy version ", scipy.version.full_version)
    # Thermodynamic data
    print ('R = {:.5g} J / mol / K'.format(thermo.R))
    print ('cd * R = {:.5g} J / mol / K'.format(thermo.cd_))
    print ('cv * R = {:.5g} J / mol / K'.format(thermo.cv_))
    print ('cl * R = {:.5g} J / mol / K'.format(thermo.cl_))
    print ('Md * R = {:.5g} J / mol / K'.format(thermo.Md_))
    print ('Mw * R = {:.5g} J / mol / K'.format(thermo.Mw_))
    print ('pc = {:.5g} Pa'.format(thermo.pc))
    print ('Tc = {:.5g} K'.format(thermo.Tc))
    print ('Lc = {:.5g} J / mol'.format(thermo.Lc * thermo.R))

    algs = ['munkres', 'divide-and-conquer', 'randallwang', 'greedy-from-top', 'lorenz', 'initial', 'munkres-worst']

    # Results
    for name in algs:
        s = '{: <24}'.format(name)

        for r in runs:
            h = r.enthalpy_range
            s += ' & '
            if name in r.results:
                s += '{0:<9.{prec}g}'.format(r.results[name].score * h, prec = prec)
            else:
                s += '{: <9}'.format('-')

        for r in runs:
            s += ' & '
            if name in r.results:
                s += '{:4.2f}'.format(r.results[name].time)
            else:
                s += '{: <5}'.format('-')
        s += ' \\\\'
        print (s)

if __name__ == "__main__":
    # Total execution time about 7 minutes on my machine
    make_figure1()
    make_figure2()
    r = compute()
    print_results(r)

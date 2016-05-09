import math
import scipy.optimize as opt

log = math.log
exp = math.exp

small   = 1e-20                 # unitless
T0      = 1                     # K
Tcrit   = 650                   # K
zero_C  = 273.15                # K
p0      = 1                     # Pa
atm     = 101325                # Pa
bar     = 100000                # Pa

# Tcrit is slightly above the critical point of water. This is used as an upperbound
# on values of T that would be even vaguely physically reasonable for our thermodynamic
# equations. Exact value here is unimportant.


#####
# The following values are quoted exactly* from
#   CRC Handbook of Chemistry and Physics, 84th edition, 2003-2004, ISBN 0-8493-0484-9
# Comments specify units and section number in the source.
# *changes in units like g -> kg are done silently
#####

R       = 8.314510              # J / mol / K           1-54
Mw_     = 0.01801528            # kg / mol              6-4     molecular weight of water

# enthalpy of vaporization of water at specified temperature
vap_T   = zero_C                # K                     6-3
vap_enthalpy = 45054            # J / mol               6-3

# heat capacity and density of air (dry air?) at specified temperature and 1 bar pressure
air_T   = 300                   # K                     6-1
air_cp  = 1007                  # J / kg / K            6-1
air_rho = 1.161                 # kg / m^3              6-1

# heat capacity of liquid water at specified temperature
lw_T    = 10 + zero_C           # K                     6-3
lw_cp   = 4192.1                # J / kg / K            6-3

# saturation vapor pressure at specified temperature
sat_T   = 10 + zero_C           # K                     6-10
sat_p_star = 1228.1             # Pa                    6-10


####
# End of CRC reference values
###

# Old value of cv_ I was using is 37.47 J / mol / K.
# New value is from the following source:
#   1870 J / kg / K (or 33.68857 J / mol / K)
#   page 77
#   Iribarne & Godson (Eds.) (2012). Atmospheric thermodynamics (Vol. 6). Springer Science & Business Media.

# Derived values

Md_     = air_rho * R * air_T / bar     # kg / mol      molecular weight of air
cd_     = air_cp * Md_                  # J / mol / K   heat capacity of air, constant pressure
cv_     = 1870 * Mw_                    # J / mol / K   heat capacity of water vapor, constant p
cl_     = lw_cp * Mw_                   # J / mol / K   heat capacity of liquid water, constant p

cd      = cd_ / R                       # unitless
cv      = cv_ / R                       # unitless
cl      = cl_ / R                       # unitless
Md      = Md_ / R                       # kg K / J
Mw      = Mw_ / R                       # kg K / J
epsilon = Mw_ / Md_                     # unitless

Lc      = vap_enthalpy / R + (cl - cv) * vap_T # K
Tc      = sat_T                         # K
pc      = sat_p_star * exp(Lc / Tc)     # Pa

# Clausius-Clapeyron relation
def compute_p_star(T):
    return pc * exp((cv - cl) * log(T / Tc) - Lc / T)

def compute_y_s(p, p_star):
    return p_star / (p - p_star)

def compute_y_s_from_T(p, T):
    return compute_y_s(p, compute_p_star(T))

def compute_ell(T):
    return cv - cl + Lc / T

def compute_issat_ypT(y, p, T):
    y_s = compute_y_s_from_T(p, T)
    return (y_s > 0) and (y > y_s)

# Correctness of this is non-trivial.
def compute_issat_yps(y, p, s):
    return compute_issat_ypT(y, p, compute_T_unsat(y, p, s))

def compute_M(y):
    return Md * (1 + epsilon * y)

def compute_Ms_unsat(y, p, T):
    if y < small:
        return cd * log(T / T0) - log(p / p0)
    else:
        return ((cd + y * cv) * log(T / T0)
                - (1 + y) * log(p / p0)
                + (1 + y) * log(1 + y)
                - y * log(y))

def compute_Ms_sat(y, p, T):
    p_star = compute_p_star(T)
    y_s = compute_y_s(p, p_star)
    ell = compute_ell(T)
    if y < small:
        # Unlikely to represent a physical situation,
        # since y > y_s for saturated parcels.
        return cd * log(T / T0) - log(p_star / p0) + log(y_s) + y_s * ell
    else:
        return ((cd + y * cv) * log(T / T0)
                - (1 + y) * log(p_star / p0)
                + log (y_s)
                + (y_s - y) * ell)

def compute_T_unsat(y, p, s):
    Ms = compute_M(y) * s
    if y < small:
        return T0 * exp((Md * s + log(p / p0)) / cd)
    else:
        return T0 * exp(
                    (Ms + (1 + y) * log(p / p0) - (1 + y) * log(1 + y) + y * log(y))
                    / (cd + y * cv)
                )

#
# For ease of writing this function and computation speed, we assume that the parcel
# specified is saturated, that y > 1e-10, that p < 1e10 Pa, and that the parcel's temperature
# is less than Tcrit. If any of these assumptions are violated this function may diverge,
# throw an exception, or return a nonsense value.
#
# This function is the main bottleneck in speeding up the code.
#
def compute_T_sat(y, p, s):
    if y < 1e-10 or p > 1e10:
        raise ValueError()

    #
    # Equation we wish to solve:
    #   M * s = (cd + y*cv) * log(T / T0) - (1 + y)*log(p_star / p0) + log(y_s) + (y_s - y) * ell
    # where
    #   p_star is a function of T
    #   y_s = p_star / (p - p_star)
    #   ell = cv - cl + Lc / T
    #
    # Note that for T < Tcrit, ell > 0 and d p_star/dT > 0.
    #
    # Let
    #   f(T) = c0 * log(T) - (1 + y) * log(p_star) + log(y_s) + (y_s - y) * ell + c1
    #       = c0 * log(T) - y * log(p_star) - log(p - p_star) + (y_s - y) * ell + c1
    #       = c0 * log(T) - y * ((cv - cl) log(T / Tc) - Lc / T) - log(p - p_star)
    #           + y_s * ell - y * (cv - cl) - y * Lc / T + c1 - y * log(pc)
    #       = c0 * log(T) - y * (cv - cl) * log(T) - log(p - p_star)
    #           + y_s * ell + c2
    #       = c3 * log(T) - log(p - p_star) + y_s * ell + c2
    # where
    #   c0 = cd + y * cv
    #   c1 = - (cd + y * cv) * log(T0) + (1 + y) * log(p0) - compute_M(y) * s
    #   c2 = c1 - y * log(pc) - y * (cv - cl) + y * (cv - cl) * log(Tc)
    #   c3 = cd + y * cl
    #
    # Note that f(T) is increasing in T for reasonable values of p and T. We want to find
    # where f(T) = 0.
    #

    c1 = - (cd + y * cv) * log(T0) + (1 + y) * log(p0) - compute_M(y) * s
    c2 = c1 - y * log(pc) - y * (cv - cl) + y * (cv - cl) * log(Tc)
    c3 = cd + y * cl

    #
    # Since the parcel is saturated we know that y_s < y, so
    #   p_star = p (y_s / (1 + y_s)) = p (1 - 1 / (1 + y_s)) < p (1 - 1 / (1 + y))
    # so we have an upperbound on the value of p_star. Furthermore, since cv - cl < 0,
    #   p_star = pc exp((cv - cl) log(T / Tc) - Lc / T)
    #       > pc exp((cv - cl) log(Tcrit / Tc) - Lc / T)
    # so
    #   -Lc / T < log(p_star / pc) + (cl - cv) log(Tcrit / Tc)
    #   Lc / T > -log(p_star / pc) + (cv - cl) log(Tcrit / Tc)      [1]
    #   T < Lc / (-log(p_star / pc) + (cv - cl) log(Tcrit / Tc))
    #   T < Lc / (-log(p / pc) - log(y / (1 + y)) + (cv - cl) log(Tcrit / Tc))
    # where we have used that the right side of [1] is positive for p_star smaller than 1e11 Pa
    # or so.
    #

    c4 = (cv - cl) * log(Tcrit / Tc)
    p_star_max = p * y / (1 + y)
    Tmax = Lc / (c4 - log(p_star_max / pc))
    Tmax = min(Tmax, Tcrit)

    # Couldn't figure out a good way to lower bound it. 100 K is pretty safe.

    Tmin = 100

    def f(T):
        p_star = compute_p_star(T)
        if p_star >= p_star_max:
            return T * 1.0e200
        y_s = p_star / (p - p_star)
        ell = cv - cl + Lc / T
        return c3 * log(T) - log(p - p_star) + y_s * ell + c2

    if f(Tmin) >= 0:
        return Tmin
    if f(Tmax) <= 0:
        return Tmax
    return opt.brentq(f, Tmin, Tmax)

def compute_Tv_sat(y, p, s):
    T = compute_T_sat(y, p, s)
    y_s = compute_y_s_from_T(p, T)
    return T * (1 + y_s) / (1 + y * epsilon)

def compute_Tv_unsat(y, p, s):
    return compute_T_unsat(y, p, s) * (1 + y) / (1 + y * epsilon)

def compute_Mh_unsat(y, p, s):
    return (cd + y * cv) * compute_T_unsat(y, p, s)

def compute_Mh_sat(y, p, s):
    T = compute_T_sat(y, p, s)
    y_s = compute_y_s_from_T(p, T)
    ell = compute_ell(T)
    return (cd + y * cv + (y_s - y) * ell) * T

def compute_Mh_dp_unsat(y, p, s):
    return (1 + y) * compute_T_unsat(y, p, s) / p

def compute_Mh_dp_sat(y, p, s):
    T = compute_T_sat(y, p, s)
    y_s = compute_y_s_from_T(p, T)
    return (1 + y_s) * T / p

##############################
#
#   User-friendly thermodynamic functions with user-friendly names
#
##############################

# w is kg / kg
def compute_w(y):
    return y * epsilon

# y is mol / mol
def compute_y(w):
    return w / epsilon

# kg / mol
def molecular_weight_water():
    return Mw_

# kg / mol
def molecular_weight_dry_air():
    return Md_

# kg / mol
def molecular_weight_moist_air(y):
    return (Md_ + y * Mw_) / (1 + y)

# partial pressure of water vapor at the saturation point
# Pa
def saturation_vapor_pressure(T):
    return p_star(T)

# unitless
def relative_humidity(y, p, T):
    y_s = compute_y_s_from_T(p, T)
    if y > y_s:
        return 1
    else:
        return y / y_s

# J / mol
def latent_heat_condensation(T):
    return compute_ell(T) * R * T

# True or False
def is_saturated(y, p, T):
    return compute_issat_ypT(y, p, T)

# J / kg / K
def entropy(y, p, T):
    if compute_issat_ypT(y, p, T):
        return compute_Ms_sat(y, p, T) / compute_M(y)
    else:
        return compute_Ms_unsat(y, p, T) / compute_M(y)

# K
def temperature(y, p, s):
    if compute_issat_yps(y, p, s):
        return compute_T_sat(y, p, s)
    else:
        return compute_T_unsat(y, p, s)

# K
def virtual_temperature(y, p, s):
    if compute_issat_yps(y, p, s):
        return compute_Tv_sat(y, p, s)
    else:
        return compute_Tv_unsat(y, p, s)

# J / kg
def enthalpy(y, p, s):
    if compute_issat_yps(y, p, s):
        return compute_Mh_sat(y, p, s) / compute_M(y)
    else:
        return compute_Mh_unsat(y, p, s) / compute_M(y)

# J / kg / Pa = m^3 / kg, units of specific volume
def enthalpy_dp(y, p, s):
    if compute_issat_yps(y, p, s):
        return compute_Mh_dp_sat(y, p, s) / compute_M(y)
    else:
        return compute_Mh_dp_unsat(y, p, s) / compute_M(y)

# For a parcel moving from pold to pnew, given the old temperature,
# compute the new temperature
# K
def new_temperature(y, Told, pold, pnew):
    return temperature(y, pnew, entropy(y, pold, Told))

# For a parcel moving from pold to pnew, given the old temperature,
# compute the change in enthalpy
# J / kg
def change_in_enthalpy(y, Told, pold, pnew):
    s = entropy(y, Told, pold)
    return enthalpy(y, pnew, s) - enthalpy(y, pold, s)

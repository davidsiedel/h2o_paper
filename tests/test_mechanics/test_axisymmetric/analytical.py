import numpy as np

from h2o.problem.problem import Problem
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.material import Material
from h2o.problem.load import Load
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *
from mgis import behaviour as mgis_bv

import matplotlib.pyplot as plt

E = 200.e6
nu = 0.4999

time_final = 1.0e-3

pi = 300.e6 * time_final
pe = 100.e6 * time_final

ri = 0.8
re = 1.0

A = (pi*(ri**2) - pe*(re**2))/(re**2 - ri**2)
B = (pi - pe)/(re**2 - ri**2)*((ri**2)*(re**2))

C = 2.0*nu*A

a = (1.0 - nu)*A/E - nu*C/E
mu = E/(2.0*(1.0+nu))
b = B/(2.0 * mu)
c = (1.0/E)*(C - 2.0*nu*A)

res_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_axisymmetric/res_incompressible/res_qdp_000009.csv"

with open(res_path, "r") as resfile:
    cf = resfile.readlines()
    r_s = []
    ur_s = []
    epsrr_s = []
    epstt_s = []
    epszz_s = []
    for line in cf[1:]:
        r = float(line.split(",")[0])
        z = float(line.split(",")[1])
        ur = float(line.split(",")[2])
        uz = float(line.split(",")[3])
        epsrr = float(line.split(",")[4]) - 0.0
        epstt = float(line.split(",")[6]) - 0.0
        epszz = float(line.split(",")[5]) - 0.0
        sigrr = float(line.split(",")[7])
        sigtt = float(line.split(",")[9])
        sigzz = float(line.split(",")[8])
        if z < 0.9259259259257181 + 1.e-4 and z > 0.9259259259257181 - 1.e-4:
            r_s.append(r)
            ur_s.append(ur)
            epsrr_s.append(epsrr)
            epstt_s.append(epstt)
            epszz_s.append(epszz)
r_a = np.linspace(ri, re, 200, endpoint=True)
ur_a = [(a * ri + b/ri) for ri in r_a]
epsrr_a = [(a - b/(ri**2)) for ri in r_a]
epstt_a = [(a + b/(ri**2)) for ri in r_a]
epszz_a = [c for ri in r_a]
plt.scatter(r_s, ur_s, label="hho u", c="grey")
plt.plot(r_a, ur_a, label="analytical u", c="grey")
plt.xlabel("position [mm]")
plt.ylabel("displacement [mm]")
plt.legend()
plt.grid()
plt.show()
plt.scatter(r_s, epsrr_s, label="hho eps_rr", c="r")
plt.plot(r_a, epsrr_a, label="analytical eps_rr", c="r")
plt.scatter(r_s, epstt_s, label="hho eps_tt", c="g")
plt.plot(r_a, epstt_a, label="analytical eps_tt", c="g")
plt.scatter(r_s, epszz_s, label="hho eps_zz", c="b")
plt.plot(r_a, epszz_a, label="analytical eps_zz", c="b")
plt.xlabel("position [mm]")
plt.ylabel("displacement gradient")
plt.legend()
plt.grid()
plt.show()
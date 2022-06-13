import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os import path, walk
kiloN: float = 1.e-3

def getCastemData(file_path: str):
    with open(file_path, "r") as resfile:
        c = resfile.readlines()
        x = []
        y = []
        for line in c[1:]:
            x_val = float(line.split(",")[1]) * 1.e3
            y_val = float(line.split(",")[2]) * kiloN
            x.append(x_val)
            y.append(y_val)
    return x, y

def getHHOData(file_path: str):
    with open(file_path, "r") as resfile:
        c = resfile.readlines()
        x = []
        y = []
        for line in c[1:]:
            sig = float(line.split(",")[0])
            disp = float(line.split(",")[2])
            radius_init = 30.e-3
            radius = radius_init - disp
            # radius = 30.e-3
            x_val = float(line.split(",")[1]) * 1.e3
            # y_val = (2.0 * np.pi * radius**2) * sig/62.0
            y_val = (np.pi * (radius**2)) * sig / radius * 1.e-3
            coef: float = 1.019e-1
            # coef: float = 1.0e-1
            y_val = radius ** 2 * sig * coef * kiloN
            # y_val = radius_init ** 2 * sig * 1.0e-1
            # y_val = radius_init ** 2 * sig * 1.0e-1
            # y_val = radius**2 * sig
            # y_val = (np.pi * (30.e-3)**2) * sig
            x.append(x_val)
            y.append(y_val)
    return x, y

def getHHOData2(file_path: str):
    with open(file_path, "r") as resfile:
        c = resfile.readlines()
        x = []
        y = []
        for line in c[1:]:
            x_val = float(line.split(",")[0])
            y_val = -float(line.split(",")[1]) * kiloN
            x.append(x_val)
            y.append(y_val)
    return x, y

# font = {'family' : 'sans-serif',
#         'weight' : 'bold',
#         'size'   : 22}
# matplotlib.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "35"
plt.figure(figsize=(10.0, 10.0))
fontsize: int = 22

xc, yc = getCastemData(
    "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening_axisymmetric/data/SSNA303_FU.csv")
xh, yh = getHHOData(
    "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening_axisymmetric/data/data_ssna.txt")
xh2, yh2 = getHHOData2(
    "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening_axisymmetric/res_hho_eq_1/output_TOP_1.csv")
plt.plot(xc, yc, label="Q2RI", color="red")
plt.scatter(xh, yh, label="HHO", s=100, color="blue", marker='o')
# plt.plot(xh2, yh2, label="HHO LAGRANGE", dashes=[2,2])
plt.xlabel("Displacement [mm]")
plt.ylabel("Force [kN]")
# plt.axis([0.0, 5.e-5, 0.0, 2.e4])
plt.legend()
plt.grid()
plt.show()
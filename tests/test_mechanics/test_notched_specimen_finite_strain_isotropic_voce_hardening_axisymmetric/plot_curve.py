import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os import path, walk

file_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening_axisymmetric/res/output_BOTTOM_1.csv"
file_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening_axisymmetric/res_hho_high/output_BOTTOM_1.csv"
castem_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening_axisymmetric/data/SSNA_Q8RI_AXI_CAST3M.csv"

with open(file_path, "r") as resfile:
    c = resfile.readlines()
    x = []
    y = []
    for line in c[1:]:
        x_val = float(line.split(",")[0])
        # y_val = float(line.split(",")[1]) / (np.pi * ((30.e-3) ** 2 ))
        y_val = float(line.split(",")[1])
        # if x_val <= 0.0003:
        x.append(x_val)
        y.append(y_val)

def get_data(file_path: str):
    with open(file_path, "r") as resfile:
        c = resfile.readlines()
        x = []
        y = []
        for line in c[1:]:
            coef = 1.0
            if "TOP" in file_path:
                coef = -1.0
            x_val = float(line.split(",")[0])
            # y_val = float(line.split(",")[1]) / (np.pi * ((30.e-3) ** 2 ))
            y_val = coef * float(line.split(",")[1])
            # if x_val <= 0.0003:
            x.append(x_val)
            y.append(y_val)
    return x, y

x_c, y_c = get_data(castem_path)
x_h, y_h = get_data(file_path)

plt.plot(x_h, y_h, label = "HHO")
plt.plot(x_c, y_c, label = "CASTEM")
plt.grid()
plt.legend()
plt.show()
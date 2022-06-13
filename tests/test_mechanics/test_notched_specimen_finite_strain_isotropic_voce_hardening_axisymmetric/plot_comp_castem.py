import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os import path, walk

lim_val = .0006

file_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening_axisymmetric/res/output_BOTTOM_1.csv"
# file_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening_axisymmetric/res_traction_force_test/output_BOTTOM_1.csv"

with open(file_path, "r") as resfile:
    c = resfile.readlines()
    x = []
    y = []
    for line in c[1:]:
        x_val = float(line.split(",")[0])
        y_val = float(line.split(",")[1])
        if x_val <= lim_val:
            x.append(x_val)
            y.append(y_val)

def extract_catsem_curve(fp):
    with open(fp, "r") as resfile:
        c = resfile.readlines()
        x = []
        y = []
        for line in c[1:]:
            x_val = float(line.split()[1])
            y_val = float(line.split()[2])
            if x_val <= lim_val:
                x.append(x_val)
                y.append(y_val)
    return x, y

file_path_qu4 = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening_axisymmetric/data/axi_qu42.csv"
file_path_qu8 = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening_axisymmetric/data/axi_qu82.csv"
file_path_qu8ri = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening_axisymmetric/data/axi_qu8ri2.csv"

x_qu4, y_qu4 = extract_catsem_curve(file_path_qu4)
x_qu8, y_qu8 = extract_catsem_curve(file_path_qu8)
x_qu8ri, y_qu8ri = extract_catsem_curve(file_path_qu8ri)

plt.plot(x, y, c="black", label="hho")
plt.plot(x_qu4, y_qu4, c="r", label="qua4")
plt.plot(x_qu8, y_qu8, c="g", label="qua8")
plt.plot(x_qu8ri, y_qu8ri, c="b", label="qua8ri")
plt.grid()
plt.legend()
plt.show()
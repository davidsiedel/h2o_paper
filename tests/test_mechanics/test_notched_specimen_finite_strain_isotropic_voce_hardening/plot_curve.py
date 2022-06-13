import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os import path, walk

file_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening_axisymmetric/res/output_BOTTOM_1.csv"
file_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_STATIC_COMP_UPG_12/output_BOTTOM_1.csv"
# file_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_STATIC_COMP_UPG_12_LARGE_DEFS_BIS/output_BOTTOM_1.csv"
file_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_PIGNET_WITH_LINEAR_HARDENING/output_BOTTOM_1.csv"
file_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_PIGNET_WITHOUT_LINEAR_HARDENING/output_BOTTOM_1.csv"
file_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_PIGNET_WITH_LINEAR_HARDENING/output_BOTTOM_1.csv"
file_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/data/output_BOTTOM_1_PIGNET_LINEAR_AND_VOCE.csv"
# file_path2 = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_STATIC_COMP_UPG_12_SMALL_DEF/output_TOP_1.csv"
catsem_res_q8ri_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/data/SSNA303_FU_WITH_LINEAR_HARDENING.csv"
catsem_res_pignet_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/data/test_FU_WITH_LINEAR_HARDENING.csv"
catsem_res_pignet_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/data/REACTION_BOTTOM_PIGNET_LINEAR_AND_VOCE_CAST3M_Q8RI.csv"
pignet_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/data/pignet.dat"
# --- SSNA
catsem_res_ssna_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/data/REACTION_BOTTOM_SSNA_LINEAR_AND_VOCE_CAST3M_Q8RI.csv"
ssna_hho_file_path = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_SSNA_WITH_LINEAR_HARDENING_HIGH_STAB_100/output_BOTTOM_1.csv"

file_path_stab_c = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_SSNA_WITH_LINEAR_HARDENING/output_BOTTOM_1.csv"
file_path_stab_0 = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/2021_11_29_comparison_Zeb_Aster_H2o/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_50.csv"
file_path_stab_1 = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/2021_11_29_comparison_Zeb_Aster_H2o/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_100.csv"
file_path_stab_2 = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/2021_11_29_comparison_Zeb_Aster_H2o/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_1000.csv"
file_path_stab_3 = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/2021_11_29_comparison_Zeb_Aster_H2o/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_5000.csv"
with open(catsem_res_ssna_path, "r") as castem_res:
    c = castem_res.readlines()
    castem_time3 = []
    castem_force3 = []
    for i, line in enumerate(c):
        # if i > 0 and i < 274:
        if i > 0:
            time = float(line.split("  ")[1])
            force = float(line.split("  ")[2])
            castem_time3.append(time)
            castem_force3.append(force)

with open(file_path_stab_c, "r") as resfile:
    c = resfile.readlines()
    x_c = []
    y_c = []
    for line in c[1:]:
        x_val = float(line.split(",")[0])
        y_val = float(line.split(",")[1])
        # if x_val <= 1.e-3:
        x_c.append(x_val)
        y_c.append(y_val)

with open(file_path_stab_0, "r") as resfile:
    c = resfile.readlines()
    x_0 = []
    y_0 = []
    for line in c[1:]:
        x_val = float(line.split(",")[0])
        y_val = float(line.split(",")[1])
        # if x_val <= 1.e-3:
        x_0.append(x_val)
        y_0.append(y_val)

with open(file_path_stab_1, "r") as resfile:
    c = resfile.readlines()
    x_1 = []
    y_1 = []
    for line in c[1:]:
        x_val = float(line.split(",")[0])
        y_val = float(line.split(",")[1])
        # if x_val <= 1.e-3:
        x_1.append(x_val)
        y_1.append(y_val)

with open(file_path_stab_2, "r") as resfile:
    c = resfile.readlines()
    x_2 = []
    y_2 = []
    for line in c[1:]:
        x_val = float(line.split(",")[0])
        y_val = float(line.split(",")[1])
        # if x_val <= 1.e-3:
        x_2.append(x_val)
        y_2.append(y_val)

with open(file_path_stab_3, "r") as resfile:
    c = resfile.readlines()
    x_3 = []
    y_3 = []
    for line in c[1:]:
        x_val = float(line.split(",")[0])
        y_val = float(line.split(",")[1])
        # if x_val <= 1.e-3:
        x_3.append(x_val)
        y_3.append(y_val)

plt.plot(castem_time3, castem_force3, label="CAST3MQ8RI")
plt.plot(x_c, y_c, label="HHO STAB 1")
plt.plot(x_0, y_0, label="HHO STAB 0.001")
plt.plot(x_1, y_1, label="HHO STAB 50")
plt.plot(x_2, y_2, label="HHO STAB 100")
plt.plot(x_3, y_3, label="HHO STAB 500")
# plt.plot(xpignet, ypignet, label="PIGNET",  c='green')
# plt.scatter(xpignet, ypignet, c='green')
plt.legend()
plt.grid()
plt.show()


castem_time3 = []
castem_force3 = []
with open(catsem_res_pignet_path, "r") as castem_res:
    c = castem_res.readlines()
    for i, line in enumerate(c):
        # if i > 0 and i < 274:
        if i > 0:
            time = float(line.split("  ")[1])
            force = float(line.split("  ")[2])
            castem_time3.append(time)
            castem_force3.append(force)

with open(file_path, "r") as resfile:
    c = resfile.readlines()
    x = []
    y = []
    for line in c[1:]:
        x_val = float(line.split(",")[0])
        y_val = float(line.split(",")[1])
        # if x_val <= 1.e-3:
        x.append(x_val)
        y.append(y_val)

with open(pignet_path, "r") as resfile:
    c = resfile.readlines()
    xpignet = []
    ypignet = []
    for line in c[1:]:
        x_val = float(line.split()[0])*1.e-3
        y_val = float(line.split()[1])*1.e6
        # if x_val <= 0.0003:
        xpignet.append(x_val)
        ypignet.append(y_val)

# with open(file_path2, "r") as resfile:
#     c = resfile.readlines()
#     x2 = []
#     y2 = []
#     for line in c[1:]:
#         x_val = float(line.split(",")[0])
#         y_val = -float(line.split(",")[1])
#         # if x_val <= 0.0003:
#         x2.append(x_val)
#         y2.append(y_val)
plt.plot(x, y, label="HHO")
# plt.plot(x2, y2)
plt.plot(castem_time3, castem_force3, label="CAST3MQ8RI")
plt.plot(xpignet, ypignet, label="PIGNET",  c='green')
plt.scatter(xpignet, ypignet, c='green')
plt.legend()
plt.grid()
plt.show()
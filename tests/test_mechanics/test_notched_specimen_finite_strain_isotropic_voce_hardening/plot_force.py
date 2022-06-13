import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os import path, walk
from typing import List
from post_processing import get_stress

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 12}
#
# matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': 14})

res_folder = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_geof"


def __get_reaction_curve(res_folder: str, num_time_steps: int):
    _, _, filenames = next(walk(res_folder))
    forces = []
    for time_step_index in range(num_time_steps):
        for filename in filenames:
            if "{}".format(time_step_index).zfill(6) in filename and "qdp" in filename:
                hho_file_path = path.join(res_folder, filename)
                with open(hho_file_path, "r") as hho_res_file:
                    index = 10459
                    c_hho = hho_res_file.readlines()
                    line = c_hho[index]
                    x_coordinates = float(line.split(",")[0])
                    y_coordinates = float(line.split(",")[1])
                    x_disp = float(line.split(",")[2])
                    sig_11 = float(line.split(",")[10])
                    force = sig_11 * ((0.0054 + x_disp))
                    print("done")
                    forces.append(force)
    return forces
    # plt.plot(time_steps, forces, label="python HHO")
    # plt.plot(cast_times, cast_forces, label="Cast3M", linestyle='--')
    # plt.legend()
    # plt.xlabel("displacement [m]")
    # plt.ylabel("reaction force [N]")
    # plt.grid()
    # plt.show()

catsem_res_path = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/SSNA303_FU.csv"
catsem_res_tri3_path = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/SSNA303_FU_tri3.csv"
catsem_res_q8ri_path = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/SSNA303_FU_Q8RI.csv"
mfem_res_linear = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/mfemmgis.csv"
mfem_res_quadratic = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/mfemmgis2.csv"
catsem_res_qua4_path = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/SSNA303_FU_qua4.csv"
castem_time = []
castem_force = []
with open(catsem_res_path, "r") as castem_res:
    c = castem_res.readlines()
    for i, line in enumerate(c):
        # if i > 0 and i < 274:
        if i > 0:
            time = float(line.split("  ")[1])
            force = float(line.split("  ")[2])
            castem_time.append(time)
            castem_force.append(force)
castem_time2 = []
castem_force2 = []
with open(catsem_res_tri3_path, "r") as castem_res:
    c = castem_res.readlines()
    for i, line in enumerate(c):
        # if i > 0 and i < 274:
        if i > 0:
            time = float(line.split("  ")[1])
            force = float(line.split("  ")[2])
            castem_time2.append(time)
            castem_force2.append(force)
castem_time3 = []
castem_force3 = []
with open(catsem_res_q8ri_path, "r") as castem_res:
    c = castem_res.readlines()
    for i, line in enumerate(c):
        # if i > 0 and i < 274:
        if i > 0:
            time = float(line.split("  ")[1])
            force = float(line.split("  ")[2])
            castem_time3.append(time)
            castem_force3.append(force)
castem_time4 = []
castem_force4 = []
with open(catsem_res_qua4_path, "r") as castem_res:
    c = castem_res.readlines()
    for i, line in enumerate(c):
        # if i > 0 and i < 274:
        if i > 0:
            time = float(line.split("  ")[1])
            force = float(line.split("  ")[2])
            castem_time4.append(time)
            castem_force4.append(force)
mfem_time = []
mfem_force = []
with open(mfem_res_linear, "r") as mfem_res:
    c = mfem_res.readlines()
    for i, line in enumerate(c):
        # if i > 0 and i < 274:
        if i >= 0:
            # print(line.split(" "))
            time = float(line.replace("\n", "").split(" ")[0]) * 8.0e-4
            force = float(line.replace("\n", "").split(" ")[2])
            print(force)
            mfem_time.append(time)
            mfem_force.append(force)
mfem_time2 = []
mfem_force2 = []
with open(mfem_res_quadratic, "r") as mfem_res:
    c = mfem_res.readlines()
    for i, line in enumerate(c):
        # if i > 0 and i < 274:
        if i >= 0:
            # print(line.split(" "))
            time = float(line.replace("\n", "").split(" ")[0]) * 8.0e-4
            force = float(line.replace("\n", "").split(" ")[2])
            print(force)
            mfem_time2.append(time)
            mfem_force2.append(force)
# hho_res_path = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_geoflocal/output_LRD.csv"
# hho_res_path = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_geoflocal2/output_LRD.csv"
# hho_res_path = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_msh/output_BOTTOM.csv"
hho_res_path = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_STATIC_TESTFORCE/output_BOTTOM.csv"
# hho_res_path = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_msh2/output_BOTTOM.csv"
hho_time = []
hho_force = []
with open(hho_res_path, "r") as hho_res:
    c = hho_res.readlines()
    for i, line in enumerate(c):
        if i > 0:
            time = float(line.split(",")[0])
            force = np.abs(float(line.split(",")[1]))
            hho_time.append(time)
            hho_force.append(force)
plt.xlabel("displacment [m]")
plt.ylabel("force [MN]")
plt.grid()
plt.plot(np.array(castem_time2), np.array(castem_force2)/1.e6, label="CAST3M TRI3", c="green", linestyle="--", linewidth=2)
plt.plot(np.array(castem_time3), np.array(castem_force3)/1.e6, label="CAST3M Q8RI", c="purple", linestyle="--", linewidth=2)
plt.plot(np.array(castem_time4), np.array(castem_force4)/1.e6, label="CAST3M QUA4", c="orange", linestyle="--", linewidth=2)
# plt.plot(np.array(castem_time), np.array(castem_force)/1.e6, label="CAST3M TRI6", c="red", linestyle="--", linewidth=3)
# plt.plot(np.array(castem_time), np.array(castem_force)/1.e6, label="MFEM LINEAR", c="red", linestyle="--", linewidth=3)
plt.scatter(np.array(hho_time), np.array(hho_force)/1.e6, label="HHO", c="blue", s=50, marker='o')
# plt.scatter(np.array(mfem_time), np.array(mfem_force)/1.e6, label="MFEM1", c="red", s=40, marker='o')
# plt.scatter(np.array(mfem_time2), np.array(mfem_force2)/1.e6, label="MFEM2", c="green", s=40, marker='o')
# a, b = get_stress("/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_msh", 0.0054, 0.03, 0.0)
# plt.scatter(np.array(b), np.array(a)/1.e6, label="HHO2", c="yellow", s=70, marker='+')
# forces = __get_reaction_curve(res_folder, len(hho_time))
# plt.plot(hho_time, forces, label="OTHER", c="green")
# plt.scatter(castem_time, castem_force, label="CAST3M", c="red")
# plt.scatter(hho_time, hho_force, label="HHO", c="blue")
plt.legend()
plt.show()
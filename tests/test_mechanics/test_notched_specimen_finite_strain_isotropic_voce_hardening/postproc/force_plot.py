import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os import path, walk
from typing import List
# from ..post_processing import get_stress

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 12}
#
# matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': 14})


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

catsem_res_qua4_path = "QUA_4.csv"
catsem_res_qua8_path = "QUA_8.csv"
catsem_res_qua8ri_path = "QUA_8_RI.csv"
# ---
castem_qua4_time = []
castem_qua4_force = []
with open(catsem_res_qua4_path, "r") as castem_res:
    c = castem_res.readlines()
    for i, line in enumerate(c):
        # if i > 0 and i < 274:
        if i > 0:
            time = float(line.split("  ")[1])
            force = float(line.split("  ")[2])
            castem_qua4_time.append(time)
            castem_qua4_force.append(force)
# ---
castem_qua8_time = []
castem_qua8_force = []
with open(catsem_res_qua8_path, "r") as castem_res:
    c = castem_res.readlines()
    for i, line in enumerate(c):
        # if i > 0 and i < 274:
        if i > 0:
            time = float(line.split("  ")[1])
            force = float(line.split("  ")[2])
            castem_qua8_time.append(time)
            castem_qua8_force.append(force)
# ---
castem_qua8ri_time = []
castem_qua8ri_force = []
with open(catsem_res_qua8ri_path, "r") as castem_res:
    c = castem_res.readlines()
    for i, line in enumerate(c):
        # if i > 0 and i < 274:
        if i > 0:
            time = float(line.split("  ")[1])
            force = float(line.split("  ")[2])
            castem_qua8ri_time.append(time)
            castem_qua8ri_force.append(force)
# ---
hho_res_path = "../res_STATIC_TESTFORCE/output_TOP.csv"
hho1_time = []
hho1_force = []
with open(hho_res_path, "r") as hho_res:
    c = hho_res.readlines()
    for i, line in enumerate(c):
        if i > 0:
            time = float(line.split(",")[0])
            force = np.abs(float(line.split(",")[1]))
            hho1_time.append(time)
            hho1_force.append(force)
# ---
hho_res_path = "../res_STATIC_TESTFORCE/output_BOTTOM.csv"
hho2_time = []
hho2_force = []
with open(hho_res_path, "r") as hho_res:
    c = hho_res.readlines()
    for i, line in enumerate(c):
        if i > 0:
            time = float(line.split(",")[0])
            force = np.abs(float(line.split(",")[1]))
            hho2_time.append(time)
            hho2_force.append(force)
plt.xlabel("displacment [m]")
plt.ylabel("force [MN]")
plt.grid()
# plt.plot(np.array(castem_time2), np.array(castem_force2)/1.e6, label="CAST3M TRI3", c="green", linestyle="--", linewidth=2)
plt.plot(np.array(castem_qua4_time), np.array(castem_qua4_force)/1.e6, label="CAST3M QUA4", c="purple", linestyle="--", linewidth=2)
plt.plot(np.array(castem_qua8_time), np.array(castem_qua8_force)/1.e6, label="CAST3M QUA8", c="blue", linestyle="--", linewidth=2)
plt.plot(np.array(castem_qua8ri_time), np.array(castem_qua8ri_force)/1.e6, label="CAST3M QUA8RI", c="cyan", linestyle="--", linewidth=2)
# plt.plot(np.array(castem_time), np.array(castem_force)/1.e6, label="CAST3M TRI6", c="red", linestyle="--", linewidth=3)
# plt.plot(np.array(castem_time), np.array(castem_force)/1.e6, label="MFEM LINEAR", c="red", linestyle="--", linewidth=3)
# plt.scatter(np.array(castem_qua8ri_time), np.array(castem_qua8ri_force)/1.e6, label="HHO", c="blue", s=50, marker='o')
plt.scatter(np.array(hho1_time), np.array(hho1_force)/1.e6, label="HHO", c="red", s=40, marker='o')
plt.scatter(np.array(hho2_time), np.array(hho2_force)/1.e6, label="HHO(1,1)", c="orange", s=40, marker='o')
# plt.plot(np.array(hho2_time), np.array(hho2_force)/1.e6, label="HHO(2,2)", c="red")
# plt.scatter(np.array(hho2_time), np.array(hho2_force)/1.e6, label="HHO", c="blue", s=40, marker='o')
# a, b = get_stress("/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_msh2", 0.0054, 0.03, 0.0)
# plt.scatter(np.array(b), np.array(a)/1.e6, label="HHO2", c="yellow", s=70, marker='+')
# forces = __get_reaction_curve(res_folder, len(hho_time))
# plt.plot(hho_time, forces, label="OTHER", c="green")
# plt.scatter(castem_time, castem_force, label="CAST3M", c="red")
# plt.scatter(hho_time, hho_force, label="HHO", c="blue")
plt.legend()
plt.show()
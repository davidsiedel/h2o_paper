import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os import path, walk

def plot_force(file_path: str, sign: bool):
    with open(file_path, "r") as resfile:
        c = resfile.readlines()
        x = []
        y = []
        for line in c[1:]:
            x_val = float(line.split(",")[0]) / 0.2
            if sign:
                y_val = float(line.split(",")[1])
            else:
                y_val = - float(line.split(",")[1])
            x.append(x_val)
            y.append(y_val)
    return x, y

x_100, y_100 = plot_force("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_100/output_U_RIGHT_0.csv", False)
x_500, y_500 = plot_force("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_500/output_U_RIGHT_0.csv", False)
x_1000, y_1000 = plot_force("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_1000/output_U_RIGHT_0.csv", False)
x_fixed, y_fixed = plot_force("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_100_FIXED_PT2/output_U_RIGHT_0.csv", False)
x_micro050, y_micro050 = plot_force("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_MICROMORPHIC_50_steps/output_U_RIGHT_0.csv", False)
x_micro100, y_micro100 = plot_force("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_MICROMORPHIC_100_steps/output_U_RIGHT_0.csv", False)
x_micro150, y_micro150 = plot_force("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_MICROMORPHIC_COMP_ZEBULON_FREE_BOUNDARY/output_U_RIGHT_0.csv", False)
x_micro1, y_micro1 = plot_force("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_MICROMORPHIC_HHO_1/output_U_RIGHT_0.csv", False)
x_micro2, y_micro2 = plot_force("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_MICROMORPHIC_HHO_2/output_U_RIGHT_0.csv", False)
x_micro3, y_micro3 = plot_force("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_MICROMORPHIC_HHO_3/output_U_RIGHT_0.csv", False)
# x2, y2 = plot_force("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_1000/output_U_LEFT_0.csv", True)
# plt.plot(x_100, y_100, label="100 steps")
# plt.plot(x_500, y_500, label="500 steps")
# plt.plot(x_1000, y_1000, label="1000 steps")
# plt.plot(x_fixed, y_fixed, label="100 steps fixed pt")
# plt.plot(x_micro050, y_micro050, label="micro 50")
# plt.plot(x_micro100, y_micro100, label="micro 100")
plt.plot(x_micro150, y_micro150, label="micro 150")
plt.plot(x_micro1, y_micro1, label="micro 1")
plt.plot(x_micro2, y_micro2, label="micro 2")
plt.plot(x_micro3, y_micro3, label="micro 3")
plt.grid()
plt.legend()
plt.xlabel("DISPLACEMENT [m]")
plt.ylabel("LOAD [N]")
# plt.plot(x2, y2, dashes=[2, 2])
plt.show()

def get_energy_data(file_path: str, sign: bool):
    with open(file_path, "r") as resfile:
        c = resfile.readlines()
        x = []
        y = []
        for line in c[1:]:
            x_val = float(line.split(",")[0])
            if sign:
                y_val = float(line.split(",")[1])
            else:
                y_val = - float(line.split(",")[1])
            x.append(x_val)
            y.append(y_val)
    return x, y

plt.close()
x_u, y_u = get_energy_data("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_MICROMORPHIC_ENERGY_output/energies_U.csv", True)
x_d, y_d = get_energy_data("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_MICROMORPHIC_ENERGY_output/energies_D_CHI.csv", True)
plt.plot([i for i in range(len(x_u))], y_u, c="blue", label="mecha")
plt.plot([i for i in range(len(x_d))], y_d, c="green", label="xhi")
# plt.plot([i for i in range(len(x_d))], [yui + ydi for yui, ydi in zip(y_u, y_d)], c="purple", label="sum")
plt.grid()
plt.legend()
plt.show()
x_plate, y_plate = get_energy_data("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_PLATE_MICROMORPHIC/output_U_TOP_1.csv", False)
plt.plot(x_plate, y_plate)
plt.grid()
plt.legend()
plt.show()

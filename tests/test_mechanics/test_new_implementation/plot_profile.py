from typing import List

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os import path, walk


class ElementPostProcess:
    elem_type: int
    elem_type2: int
    elem_nodes: List[int]
    elem_coordinates: List[np.ndarray]

    def __init__(self,
                 a_elem_type: int,
                 a_elem_type2: int,
                 a_elem_nodes: List[int],
                 a_elem_coordinates: List[np.ndarray]
                 ):
        self.elem_type = a_elem_type
        self.elem_type2 = a_elem_type2
        self.elem_nodes = a_elem_nodes
        self.elem_coordinates = a_elem_coordinates


class FieldPostProcess:
    elem_index: int
    values: List[float]

    def __init__(self,
                 a_elem_index: int,
                 a_values: List[float]
                 ):
        self.elem_index = a_elem_index
        self.values = a_values


class FieldPostProcess2:
    elements: List[ElementPostProcess]
    field: List[FieldPostProcess]

    def __init__(self,
                 a_file: str,
                 a_label: str,
                 a_time: float
                 ):
        self.elements, offset = get_elems(a_file)
        self.field = get_field_values(a_file, self.elements, offset, a_label, a_time)


def get_elems(
        file_path: str
):
    with open(file_path, "r") as resfile:
        c = resfile.readlines()
        offset: int = 4
        num_nodes: int = int(c[offset])
        nodes: np.ndarray = np.zeros((3, num_nodes))
        offset += 1
        for line_count in range(offset, num_nodes + offset):
            node_index: int = line_count - offset
            x: float = float(c[line_count].split()[1])
            y: float = float(c[line_count].split()[2])
            z: float = float(c[line_count].split()[3])
            node: np.ndarray = np.array([x, y, z])
            nodes[:, node_index] = node
        offset += num_nodes + 2
        num_elems: int = int(c[offset])
        offset += 1
        elems: List[ElementPostProcess] = []
        for line_count in range(offset, num_elems + offset):
            elem_index: int = int(c[line_count].split()[0])
            elem_type: int = int(c[line_count].split()[1])
            elem_type2: int = int(c[line_count].split()[3])
            elem_nodes: List[int] = []
            for i in range(5, len(c[line_count].split())):
                elem_nodes.append(int(c[line_count].split()[i]))
            nodes_coords: List[np.ndarray] = []
            for node_index in elem_nodes:
                nodes_coords.append(nodes[:, node_index - 1])
            elems.append(ElementPostProcess(elem_type, elem_type2, elem_nodes, nodes_coords))
        offset += num_elems + 1
        return elems, offset


def get_field_values(
        file_path: str,
        a_elems: List[ElementPostProcess],
        a_offset: int,
        label: str,
        time: float
):
    elems: List[ElementPostProcess] = a_elems
    offset: int = a_offset
    found: bool = False
    field_values: List[FieldPostProcess] = []
    with open(file_path, "r") as resfile:
        c = resfile.readlines()
        while not found:
            field_label: str = c[offset + 2].replace("\"", "").replace("\n", "")
            field_time: float = float(c[offset + 4])
            num_values: int = int(c[offset + 8])
            num_components: int = int(c[offset + 7])
            temp_offset: int = 9
            print("field_label :", field_label)
            print("field_time :", field_time)
            print("num_values :", num_values)
            print("num_components :", num_components)
            if field_time == time and field_label == label:
                values: np.ndarray = np.zeros((num_components, num_values))
                for i in range(num_values):
                    elem_index: int = int(c[offset + temp_offset + i].split()[0])
                    vals: List[float] = []
                    for j in range(num_components):
                        vals.append(float(c[offset + temp_offset + i].split()[j + 1]))
                    values[:, i] = np.array(vals)
                    fpd: FieldPostProcess = FieldPostProcess(elem_index, vals)
                    field_values.append(fpd)
                return field_values
            else:
                offset += num_values + temp_offset + 1


file_path: str = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_MICROMORPHIC_HHO_1_BC_DIRICHLET/output.msh"
fp1: FieldPostProcess2 = FieldPostProcess2(
    file_path,
    "INTERNAL_VARIABLES_D_CHI",
    0.2
)
xs1, ys1 = [], []
y_max: float = 0.0
for i in range(len(fp1.field)):
    coord_y = fp1.elements[fp1.field[i].elem_index - 1].elem_coordinates[0][1]
    if coord_y > y_max:
        y_max = coord_y
for i in range(len(fp1.field)):
    y = fp1.field[i].values[0]
    coord_x = fp1.elements[fp1.field[i].elem_index - 1].elem_coordinates[0][0]
    coord_y = fp1.elements[fp1.field[i].elem_index - 1].elem_coordinates[0][1]
    eps: float = 1.e-2
    if y_max - eps < coord_y < y_max + eps and 0.0 < coord_x < 1.00001:
        xs1.append(coord_x)
        ys1.append(y)

file_path: str = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_MICROMORPHIC_HHO_2_BC_DIRICHLET/output.msh"
fp2: FieldPostProcess2 = FieldPostProcess2(
    file_path,
    "INTERNAL_VARIABLES_D_CHI",
    0.2
)
xs2, ys2 = [], []
y_max: float = 0.0
for i in range(len(fp2.field)):
    coord_y = fp2.elements[fp2.field[i].elem_index - 1].elem_coordinates[0][1]
    if coord_y > y_max:
        y_max = coord_y
for i in range(len(fp2.field)):
    y = fp2.field[i].values[0]
    coord_x = fp2.elements[fp2.field[i].elem_index - 1].elem_coordinates[0][0]
    coord_y = fp2.elements[fp2.field[i].elem_index - 1].elem_coordinates[0][1]
    eps: float = 1.e-2
    if y_max - eps < coord_y < y_max + eps and 0.0 < coord_x < 1.00001:
        xs2.append(coord_x)
        ys2.append(y)

file_path: str = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_MICROMORPHIC_HHO_2/output.msh"
fp3: FieldPostProcess2 = FieldPostProcess2(
    file_path,
    "INTERNAL_VARIABLES_D_CHI",
    0.2
)
xs3, ys3 = [], []
y_max: float = 0.0
for i in range(len(fp3.field)):
    coord_y = fp3.elements[fp3.field[i].elem_index - 1].elem_coordinates[0][1]
    if coord_y > y_max:
        y_max = coord_y
for i in range(len(fp3.field)):
    y = fp3.field[i].values[0]
    coord_x = fp3.elements[fp3.field[i].elem_index - 1].elem_coordinates[0][0]
    coord_y = fp3.elements[fp3.field[i].elem_index - 1].elem_coordinates[0][1]
    eps: float = 1.e-2
    if y_max - eps < coord_y < y_max + eps and 0.0 < coord_x < 1.00001:
        xs3.append(coord_x)
        ys3.append(y)
# plt.plot(xs1, ys1, label="HHO-uf1,uc1,df1,dc0")
# plt.plot(xs2, ys2, label="HHO2-uf2,uc2,df2,dc1")
plt.plot(xs1, ys1, label="HHO(0)")
plt.plot(xs2, ys2, label="HHO(1)")
# plt.plot(xs3, ys3, dashes=[2,2], label="HHO3-uf3,uc3,df3,dc2")
plt.axis([0.4, 0.5, 0.0, 1.0])
plt.legend()
plt.xlabel("POSITION [mm]")
plt.ylabel("DAMAGE")
plt.grid()
plt.show()

file_path: str = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ROD_100_FIXED_PT2_SHEAR_STRESS_NO_ZERO_DAMAGE2/output.msh"
fp4: FieldPostProcess2 = FieldPostProcess2(
    file_path,
    "NODE_FIELD_D",
    0.013
)
xs4, ys4 = [], []
y_max: float = 0.0
for i in range(len(fp4.field)):
    coord_y = fp4.elements[fp4.field[i].elem_index - 1].elem_coordinates[0][1]
    if coord_y > y_max:
        y_max = coord_y
for i in range(len(fp4.field)):
    y = fp4.field[i].values[0]
    coord_x = fp4.elements[fp4.field[i].elem_index - 1].elem_coordinates[0][0]
    coord_y = fp4.elements[fp4.field[i].elem_index - 1].elem_coordinates[0][1]
    eps: float = 1.e-2
    if y_max - eps < coord_y < y_max + eps and 0.0 - eps < coord_x < 1.00001:
        xs4.append(coord_x)
        ys4.append(y)

young_modulus: float = 210000.0
l_0: float = 1.0
g_c: float = 2.7
ell: float = 0.05
u_s: float = 0.010088
# ell: float = 0.1
si_m = 3./16. * np.sqrt((3.0 * young_modulus * g_c) / ell)
si_m = (3. * np.sqrt(3.0))/(8. * np.sqrt(2.0)) * np.sqrt((young_modulus * g_c) / (2.0 * ell))
u_max: float = 16. * si_m * l_0 / (9. * young_modulus)
H_energy = si_m * u_s * 0.5
d_c: float = u_s ** 2 / (u_s ** 2 + g_c/(ell * young_modulus))
lam: float = np.sqrt(1/ell**2 + young_modulus * u_s ** 2 / (g_c * ell))
C = H_energy
print("H_energy : {}".format(H_energy))
B = 1./(-np.exp(lam * l_0 / 2) + np.exp(-lam * l_0 / 2)) * (np.exp(lam * l_0 / 2) * (H_energy - 1.) + d_c - H_energy)
A = 1. - (B + H_energy)
print("B : {}".format(B))
print("A : {}".format(A))
print("A + B + C : {}".format(A * np.exp(lam * l_0 / 2) + B * np.exp(-lam * l_0 / 2) + C))
print("A + B + C : {}".format(A + B + C))
print(u_max)
# u_max: float = 0.11
print(si_m)
# B = 0.716454
# C = 0.2835
# B = np.exp(lam/2.) * (d_c -1.) / (np.exp(-lam/2.) - np.exp(lam/2.))
# B = 1 - d_c
# u_s: float = 0.0075
# u_s: float = u_max
# print("dc : {}".format(d_c))
# print("B : {}".format(B))
# lam: float = np.sqrt(1/ell**2 + (young_modulus * u_s ** 2) / (g_c * ell))
values_botom = [xi for xi in np.linspace(0., 0.5, 500, endpoint=True)]
# values_anal = [B * np.exp(-lam * np.abs(xi - 0.5)) + d_c for xi in values_botom]
values_anal = [A * np.exp(lam * np.abs(xi - 0.5)) + B * np.exp(-lam * np.abs(xi - 0.5)) + C for xi in values_botom]
plt.plot(values_botom, values_anal, label="ANALYTICAL", color="r")
plt.scatter(xs4, ys4, label="HHO", color="b")
# plt.plot(xs3, ys3, dashes=[2,2], label="HHO3-uf3,uc3,df3,dc2")
# plt.axis([0.4, 0.5, 0.0, 1.0])
plt.legend()
plt.xlabel("POSITION [mm]")
plt.ylabel("DAMAGE")
plt.grid()
plt.show()
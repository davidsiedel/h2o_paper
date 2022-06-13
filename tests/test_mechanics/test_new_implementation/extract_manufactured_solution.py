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


file_path: str = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_DAMAGE_ANALYTICAL/output.msh"
fp1: FieldPostProcess2 = FieldPostProcess2(
    file_path,
    "NODE_FIELD_D",
    1.0
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
    eps: float = 1.e-8
    # if y_max - eps < coord_y < y_max + eps and 0.0 < coord_x < 1.00001:
    xs1.append(coord_x)
    ys1.append(y)
x_analytical, y_analytical = [], []
for i in np.linspace(0.0, 0.5, 800):
    x_analytical.append(i)
    y_analytical.append(0.5 * np.sin(4.0 * np.pi * i))
plt.scatter(xs1, ys1, label="HHO GAUSS POINT", color="green")
plt.plot(x_analytical, y_analytical, label="ANALYTICAL", color="blue")
plt.xlabel("POSITION")
plt.ylabel("DAMAGE")
plt.legend()
plt.grid()
plt.show()
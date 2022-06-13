from enum import Enum, auto

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os import path, walk
from typing import List, Dict


class NodeType(Enum):
    VERTEX = auto()
    QUADRATURE = auto()


class Node:
    coordinates: np.ndarray
    tag: int
    data: List[Dict[str, np.ndarray]]
    node_type: NodeType

    def __init__(self,
                 coords: np.ndarray,
                 t: int,
                 typ: NodeType):
        self.coordinates = coords
        self.tag = t
        self.node_type = typ
        self.data = []


def get_data(file_path: str):
    with open(file_path, "r") as data_file:
        lines = data_file.readlines()
        line_count: int = 0
        for line_index, line in enumerate(lines):
            if "$Nodes" in line:
                line_count = line_index
                break
        line_count += 1
        num_nodes = int(lines[line_count])
        nodes: np.ndarray = np.zeros((3, num_nodes))
        line_count += 1
        node_count: int = 0
        for i in range(line_count, line_count + num_nodes):
            x: float = float(lines[i].split(" ")[1])
            y: float = float(lines[i].split(" ")[2])
            z: float = float(lines[i].split(" ")[3])
            nodes[:, node_count] = np.array([x, y, z])
            node_count += 1
        print(nodes)
        for line_index, line in enumerate(lines):
            if "$Elements" in line:
                line_count = line_index
                break
        line_count += 1
        num_elements = int(lines[line_count])
        line_count += 1
        element_count: int = 0
        node_list = {}
        for i in range(line_count, line_count + num_elements):
            elem_typ: int = int(lines[i].split(" ")[1])
            if elem_typ == 15:
                node_tag: int = int(lines[i].split(" ")[-1]) - 1
                typ: int = int(lines[i].split(" ")[3])
                node_type: NodeType = NodeType.VERTEX
                if typ == 0:
                    node_type = NodeType.VERTEX
                elif typ == 1:
                    node_type = NodeType.QUADRATURE
                node: Node = Node(nodes[:, node_tag], node_tag, node_type)
                node_list[node_tag] = node
                element_count += 1
            print(node_list[node_tag].node_type)
        for line_index, line in enumerate(lines):
            if "$NodeData" in line:
                line_count = line_index
                break
        line_count += 2
        field_label: str = lines[line_count]
        line_count += 4
        time: int = int(lines[line_count])
        line_count += 1
        num_components: int = int(lines[line_count])
        line_count += 1
        num_elems: int = int(lines[line_count])
        line_count += 1
        for i in range(num_elems):
            line_fetch: int = line_count + i
            elem_tag: int = int(lines[line_fetch].split(" ")[0]) - 1
            data: np.ndarray = np.zeros((len(lines[line_fetch].split(" ")),))
            for j in range(1, len(lines[line_fetch].split(" "))):
                data[j - 1] = float(lines[line_fetch].split(" ")[j])
            node_list[elem_tag].data.append({field_label: data})
        # print(node_list[0].data[0]["FLUX_U"])
        print(node_list[595].data)

data_file_path = "/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening_axisymmetric/data/analytical.csv"

def plot_data(file_path: str):
    with open(file_path, "r") as data_file:
        lines = data_file.readlines()
        x = []
        y = []
        for i, line in enumerate(lines):
            x_i = float(lines[i].split(" ")[0])
            y_i = float(lines[i].split(" ")[3])
            x.append(x_i)
            y.append(y_i)
    return x, y
    # plt.plot(x, y)
    # plt.show()

def plot_analytical():
    l_0: float = 0.1
    d_c: float = 0.0
    x_d: float = 0.5
    lam_0: float = np.sqrt((1.0/(l_0)**2))
    B: float = (np.exp(lam_0/2.0) * (d_c - 1)) / (np.exp(-lam_0/2.0) - np.exp(lam_0/2.0))
    A: float = 1.0 - (B + d_c)
    C: float = d_c
    plot_fun = lambda arg: A * np.exp(lam_0(arg - 1.0)) + B * np.exp(-lam_0(arg - 1.0)) + C
    x = []
    y = []
    for x_i in np.linspace(0.0, 1.0, 800, endpoint=True):
        # print(float(x_i))
        # y_i = plot_fun(float(x_i))
        y_i: float = A * np.exp(lam_0 * np.abs(x_i - x_d)) + B * np.exp(-lam_0 * np.abs(x_i - x_d)) + C
        # y_i: float = np.exp(-np.abs(x_i)/l_0)
        x.append(x_i)
        y.append(y_i)
    return x, y

x_h, y_h = plot_data(data_file_path)
x_a, y_a = plot_analytical()
plt.scatter(x_h, y_h, label="HHO GAUSS POINTS", c="green")
# plt.plot(x_a, y_a, label="ANALYTICAL", c="blue", dashes=[2,2])
plt.plot(x_a, y_a, label="ANALYTICAL", c="blue")
plt.xlabel("POSITION")
plt.ylabel("DAMAGE")
plt.grid()
plt.legend()
plt.show()

def check_H_fun():
    G_c: float = 1.0
    l_0: float = 0.1
    puls: float = 4.0
    x = []
    y = []
    for xi in np.linspace(0.0, 0.5, 800, endpoint=True):
        v0: float = (G_c / (2.0 * l_0)) * (1.0 / (1.0 - 0.5 * np.sin(puls * np.pi * xi)))
        v1: float = (0.5 * np.sin(puls * np.pi * xi) * (1.0 + (puls * np.pi * l_0) ** 2))
        H_v: float = v0 * v1
        x.append(xi)
        y.append(H_v)
    print(np.max(y))
    print(np.min(y))
    plt.plot(x, y)
    plt.grid()
    plt.show()
# check_H_fun()






plot_data(data_file_path)
# plot_data()
# get_data("/home/dsiedel/projects2/h2o/tests/test_mechanics/test_new_implementation/res_ANALYTICAL/output.msh")

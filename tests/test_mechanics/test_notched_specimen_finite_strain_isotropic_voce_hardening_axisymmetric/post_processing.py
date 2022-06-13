import numpy as np
from numpy import ndarray
from typing import List

def get_quadrature_node_tag(res_folder: str, pos_x: float, pos_y: float, pos_z: float) -> (int, float, float):
    nodes_data: bool = False
    elemns_data: bool = False
    nodes_line_start: int = 0
    elems_line_start: int = 0
    nodes_coordinates = []
    distance: float = np.inf
    node_tag: int = 0
    with open(res_folder + "/output.msh", "r") as outfile:
        c = outfile.readlines()
        for i, line in enumerate(c):
            if "$Nodes" in line:
                nodes_line_start = i + 1
                nodes_data = True
            if "$EndNodes" in line:
                nodes_data = False
            if "$Elements" in line:
                elems_line_start = i + 2
                elemns_data = True
            if "$EndElements" in line:
                elemns_data = False
            if i > nodes_line_start and nodes_data:
                x: float = float(line.split(" ")[1])
                y: float = float(line.split(" ")[2])
                z: float = float(line.split(" ")[3])
                node_coordinates = np.array([x, y, z])
                nodes_coordinates.append(node_coordinates)
            if i >= elems_line_start and elemns_data:
                tag: int = int(line.split(" ")[0])
                if int(line.split(" ")[3]) == 1:
                    xn: float = nodes_coordinates[tag - 1][0]
                    yn: float = nodes_coordinates[tag - 1][1]
                    zn: float = nodes_coordinates[tag - 1][2]
                    dist: float = np.sqrt((xn - pos_x) ** 2 + (yn - pos_y) ** 2 + (zn - pos_z) ** 2)
                    if dist < distance:
                        distance = dist
                        node_tag = tag
                        x_coord, y_coord, z_coord = xn, yn, zn
    return node_tag, x_coord, y_coord, z_coord


def get_stress(res_folder: str, pos_x: float, pos_y: float, pos_z: float) -> List[float]:
    tagf, x, y, z = get_quadrature_node_tag(res_folder, pos_x, pos_y, pos_z)
    out_data: bool = False
    stress_flag: bool = False
    disp_flag: bool = False
    local_index: int = 0
    node_data_index: int = 0
    stress_datas = []
    disp_dataxs = []
    disp_datays = []
    forces = []
    times = []
    with open(res_folder + "/output.msh", "r") as outfile:
        c = outfile.readlines()
        for i, line in enumerate(c):
            if "$NodeData" in line:
                out_data = True
            if "$EndNodeData" in line:
                out_data = False
                node_data_index += 1
                local_index = 0
            if out_data:
                if local_index == 2:
                    if "CAUCHY_STRESS" in line:
                        stress_flag = True
                        disp_flag = False
                    elif "QUADRATURE_DISPLACEMENT" in line:
                        stress_flag = False
                        disp_flag = True
                    else:
                        stress_flag = False
                        disp_flag = False
                if local_index > 7 and stress_flag:
                    if int(line.split(" ")[0]) == tagf:
                        # print(line)
                        stress_data: float = float(line.split(" ")[2])
                        stress_datas.append(stress_data)
                if local_index > 7 and disp_flag:
                    if int(line.split(" ")[0]) == tagf:
                        disp_datax: float = float(line.split(" ")[1])
                        disp_datay: float = float(line.split(" ")[2])
                        disp_dataxs.append(disp_datax)
                        disp_datays.append(disp_datay)
                local_index += 1
        for ss, ds, dy in zip(stress_datas, disp_dataxs, disp_datays):
            # force = ss * (x + ds)
            force = ss * (0.0054 + ds)
            forces.append(force)
            times.append(dy)
    return forces, times

# tagf, G = get_stress("/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/res_msh", 0.0054, 0.03, 0.0)
# print(tagf)
# print(G)
# import matplotlib.pyplot as plt
# plt.plot(G, tagf)
# plt.show()
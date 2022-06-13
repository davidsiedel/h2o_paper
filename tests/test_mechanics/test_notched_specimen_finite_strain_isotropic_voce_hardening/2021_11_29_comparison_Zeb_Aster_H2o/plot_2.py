from typing import List
from enum import Enum, auto
import matplotlib.pyplot as plt
import numpy as np
from os import path, walk

prefix: str = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/2021_11_29_comparison_Zeb_Aster_H2o/"
# "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/2021_11_29_comparison_Zeb_Aster_H2o/H2O_VOCE_WITH_LINEAR_HARDENING"
# /home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/2021_11_29_comparison_Zeb_Aster_H2o/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_0_dot_001.csv


class CodeName(Enum):
    ASTER = auto()
    ZEBULON = auto()
    CAST3M = auto()
    H2O = auto()
    NICOLAS = auto()


def get_curve_data(
        code_name: CodeName,
        file_path: str,
        x_col: int,
        y_col: int
) -> List[List[float]]:
    with open(file_path, "r") as file_res:
        c = file_res.readlines()
        x_list: List[float] = []
        y_list: List[float] = []
        if code_name == CodeName.ZEBULON:
            split_char: str = "  "
        elif code_name == CodeName.CAST3M:
            split_char: str = ","
        elif code_name == CodeName.NICOLAS:
            split_char: str = "  "
        elif code_name == CodeName.H2O:
            split_char: str = ","
        else:
            raise KeyError("no such code name")
        for i, line in enumerate(c):
            if i > 0:
                if code_name == CodeName.NICOLAS:
                    x_item: float = 0.001 * float(line.split(split_char)[x_col])
                    y_item: float = 1.e6 * float(line.split(split_char)[y_col])
                else:
                    x_item: float = float(line.split(split_char)[x_col])
                    y_item: float = float(line.split(split_char)[y_col])
                x_list.append(x_item)
                y_list.append(y_item)
    return [x_list, y_list]

plt.plot(
    get_curve_data(CodeName.NICOLAS, prefix + "/NICOLAS/pignet.dat", 0, 1)[0],
    get_curve_data(CodeName.NICOLAS, prefix + "/NICOLAS/pignet.dat", 0, 1)[1],
    label="NICOLAS HHO/Q8"
)
# plt.plot(
#     get_curve_data(CodeName.CAST3M, prefix + "/CATS3M/REACTION_BOTTOM_PIGNET_VOCE_CAST3M_Q8RI.csv", 1, 2)[0],
#     get_curve_data(CodeName.CAST3M, prefix + "/CATS3M/REACTION_BOTTOM_PIGNET_VOCE_CAST3M_Q8RI.csv", 1, 2)[1],
#     label="CAST3M"
# )
plt.plot(
    get_curve_data(CodeName.CAST3M, prefix + "/CATS3M/REACTION_BOTTOM_PIGNET_LINEAR_AND_VOCE_CAST3M_Q8RI.csv", 1, 2)[0],
    get_curve_data(CodeName.CAST3M, prefix + "/CATS3M/REACTION_BOTTOM_PIGNET_LINEAR_AND_VOCE_CAST3M_Q8RI.csv", 1, 2)[1],
    label="CAST3M Q8RI"
)
plt.plot(
    get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_PIGNET_LINEAR_AND_VOCE.csv", 0, 1)[0],
    get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_PIGNET_LINEAR_AND_VOCE.csv", 0, 1)[1],
    label="H2O 1"
)
plt.plot(
    get_curve_data(CodeName.ZEBULON, prefix + "/ZEBULON_VOCE_LIN/pignet_4_2_calc.test", 0, 1)[0],
    get_curve_data(CodeName.ZEBULON, prefix + "/ZEBULON_VOCE_LIN/pignet_4_2_calc.test", 0, 1)[1],
    label="ZEBULON Q8RI"
)
plt.xlabel("displacement [m]")
plt.ylabel("force [N]")
plt.grid()
plt.legend()
plt.show()

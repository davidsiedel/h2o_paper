from typing import List
from enum import Enum, auto
import matplotlib.pyplot as plt
import numpy as np
from os import path, walk

prefix: str = "/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/2021_11_29_comparison_Zeb_Aster_H2o/"
"/home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/2021_11_29_comparison_Zeb_Aster_H2o/H2O_VOCE_WITH_LINEAR_HARDENING"
# /home/dsiedel/projects2/h2o/tests/test_mechanics/test_notched_specimen_finite_strain_isotropic_voce_hardening/2021_11_29_comparison_Zeb_Aster_H2o/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_0_dot_001.csv


class CodeName(Enum):
    ASTER = auto()
    ZEBULON = auto()
    CAST3M = auto()
    H2O = auto()


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
        factor: float = 1.0
        if code_name == CodeName.ZEBULON:
            split_char: str = "  "
        elif code_name == CodeName.CAST3M:
            split_char: str = ","
        elif code_name == CodeName.H2O:
            split_char: str = ","
            factor = -1.0
        else:
            raise KeyError("no such code name")
        for i, line in enumerate(c):
            if i > 0:
                x_item: float = float(line.split(split_char)[x_col])
                y_item: float = factor * float(line.split(split_char)[y_col])
                x_list.append(x_item)
                y_list.append(y_item)
    return [x_list, y_list]


plt.plot(
    get_curve_data(CodeName.CAST3M, prefix + "/CATS3M/REACTION_BOTTOM_SSNA_LINEAR_AND_VOCE_CAST3M_Q8RI.csv", 1, 2)[0],
    get_curve_data(CodeName.CAST3M, prefix + "/CATS3M/REACTION_BOTTOM_SSNA_LINEAR_AND_VOCE_CAST3M_Q8RI.csv", 1, 2)[1],
    label="CAST3M",
    linewidth=12.0
)
plt.plot(
    get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_TOP_1.csv", 0, 1)[0],
    get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_TOP_1.csv", 0, 1)[1],
    label="H2O 1"
)
# plt.plot(
#     get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_0_dot_001.csv", 0, 1)[0],
#     get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_0_dot_001.csv", 0, 1)[1],
#     label="H2O 0.001"
# )
plt.plot(
    get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_TOP_1_50.csv", 0, 1)[0],
    get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_TOP_1_50.csv", 0, 1)[1],
    label="H2O 50"
)
# plt.plot(
#     get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_100.csv", 0, 1)[0],
#     get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_100.csv", 0, 1)[1],
#     label="H2O 100"
# )
# plt.plot(
#     get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_500.csv", 0, 1)[0],
#     get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_500.csv", 0, 1)[1],
#     label="H2O 500"
# )
# plt.plot(
#     get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_1000.csv", 0, 1)[0],
#     get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_1000.csv", 0, 1)[1],
#     label="H2O 1000"
# )
# plt.plot(
#     get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_5000.csv", 0, 1)[0],
#     get_curve_data(CodeName.H2O, prefix + "/H2O_VOCE_WITH_LINEAR_HARDENING/output_BOTTOM_1_stab_5000.csv", 0, 1)[1],
#     label="H2O 5000"
# )
plt.plot(
    get_curve_data(CodeName.ZEBULON, prefix + "/ZEBULON_VOCE_LIN/ssna303_calc.test", 0, 1)[0],
    get_curve_data(CodeName.ZEBULON, prefix + "/ZEBULON_VOCE_LIN/ssna303_calc.test", 0, 1)[1],
    label="ZEBULON Q8"
)
plt.xlabel("time")
plt.ylabel("force")
plt.legend()
plt.show()

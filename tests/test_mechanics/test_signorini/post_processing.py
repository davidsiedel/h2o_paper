import matplotlib.pyplot as plt
import numpy as np

file_path = "/home/dsiedel/projetcs/h2o/tests/test_mechanics/test_signorini/res_fine2/output_BOTTOM_0.csv"

with open(file_path, "r") as resfile:
    c = resfile.readlines()
    xs, ys = [], []
    for line in c[1:]:
        data = line.split(",")
        x = float(data[0])
        y = float(data[1])
        xs.append(x)
        ys.append(y)
    plt.xlabel("DISPLACEMENT [mm]")
    plt.ylabel("LOAD [N]")
    plt.plot(xs, ys)
    # plt.scatter(xs, ys)
    plt.grid()
    plt.show()
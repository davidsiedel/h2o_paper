import os
from mgis import behaviour as mgis_bv 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.special import binom

from matplotlib import rc

fontsize = 14
rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 
                                'monospace': ['Computer Modern Typewriter']})
params = {'backend': 'pdf',
            'font.size' : fontsize,
          'axes.labelsize': fontsize,
          'legend.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'text.usetex': True,
          'axes.unicode_minus': True}
matplotlib.rcParams.update(params)

computation_parameters = [
    # --- ORDER 1
    # CONSISTENT JACOBIAN
    {"mesh" : "coarse", "order": 1, "algorithm" : "Explicit", "jacobian" : "Consistent", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 1, "algorithm" : "Implicit", "jacobian" : "Consistent", "acceleration": "NonAccelerated"},
    # ELASTIC JACOBIAN
    {"mesh" : "coarse", "order": 1, "algorithm" : "Explicit", "jacobian" : "Elastic", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 1, "algorithm" : "Explicit", "jacobian" : "Elastic", "acceleration": "Accelerated"},
    {"mesh" : "coarse", "order": 1, "algorithm" : "Implicit", "jacobian" : "Elastic", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 1, "algorithm" : "Implicit", "jacobian" : "Elastic", "acceleration": "Accelerated"},
    # --- ORDER 2
    # CONSISTENT JACOBIAN
    {"mesh" : "coarse", "order": 2, "algorithm" : "Explicit", "jacobian" : "Consistent", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 2, "algorithm" : "Implicit", "jacobian" : "Consistent", "acceleration": "NonAccelerated"},
    # ELASTIC JACOBIAN
    {"mesh" : "coarse", "order": 2, "algorithm" : "Explicit", "jacobian" : "Elastic", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 2, "algorithm" : "Explicit", "jacobian" : "Elastic", "acceleration": "Accelerated"},
    {"mesh" : "coarse", "order": 2, "algorithm" : "Implicit", "jacobian" : "Elastic", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 2, "algorithm" : "Implicit", "jacobian" : "Elastic", "acceleration": "Accelerated"},
    # --- ORDER 3
    # CONSISTENT JACOBIAN
    {"mesh" : "coarse", "order": 3, "algorithm" : "Explicit", "jacobian" : "Consistent", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 3, "algorithm" : "Implicit", "jacobian" : "Consistent", "acceleration": "NonAccelerated"},
    # ELASTIC JACOBIAN
    {"mesh" : "coarse", "order": 3, "algorithm" : "Explicit", "jacobian" : "Elastic", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 3, "algorithm" : "Explicit", "jacobian" : "Elastic", "acceleration": "Accelerated"},
    {"mesh" : "coarse", "order": 3, "algorithm" : "Implicit", "jacobian" : "Elastic", "acceleration": "NonAccelerated"},
    {"mesh" : "coarse", "order": 3, "algorithm" : "Implicit", "jacobian" : "Elastic", "acceleration": "Accelerated"},
    # LOCAL ACCELERATION
]

algo_dict = {
    "Explicit" : {"label": "Static condensation", "linestyle": "-", "color": 'b'},
    "Implicit" : {"label": "Cell equilibrium", "linestyle": "--", "color": 'r'},
}

integration_dict = {
    "Consistent" : {"label": "consistent jacobian", "linestyle": '--', "marker": 'p'},
    "Elastic" : {"label": "elastic jacobian", "linestyle": '-', "marker": ''}
}

acceleration_dict = {
    "NonAccelerated" : {"label": "non-accelerated", "linestyle" : '-'},
    "Accelerated" : {"label": "accelerated", "linestyle" : '--'},
}

label_tag_rule = {
    "ExplicitConsistentNonAccelerated" :    {"label": "Sc", "color" : 'b', "linestyle": '--', "marker": 'o', "markevery": 5},
    "ImplicitConsistentNonAccelerated" :    {"label": "Cc", "color" : 'b', "linestyle": '--', "marker": 's', "markevery": 7},
    "ExplicitElasticNonAccelerated" :       {"label": "Se", "color" : 'r', "linestyle": '--', "marker": 'o', "markevery": 5},
    "ImplicitElasticNonAccelerated" :       {"label": "Ce", "color" : 'm', "linestyle": '--', "marker": 's', "markevery": 7},
    "ExplicitElasticAccelerated" :          {"label": "Sa", "color" : 'c', "linestyle": '--', "marker": 'o', "markevery": 5},
    "ImplicitElasticAccelerated" :          {"label": "Ca", "color" : 'g', "linestyle": '--', "marker": 's', "markevery": 7},
}

def fetch_data(batch, specimen, index, norm_index):
    c_norm = computation_parameters[norm_index]
    file_path = "/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/res{}_{}__{}__hho{}__{}__{}__{}/output.txt".format(batch, specimen, c_norm["mesh"], c_norm["order"], c_norm["algorithm"], c_norm["jacobian"], c_norm["acceleration"])
    with open(file_path, 'r') as file_n:
        iterations = []
        lines = file_n.readlines()
        for line in lines:
            if "+ ITERATIONS : " in line:
                num_iterations = line.split()[3]
                if len(iterations) == 0:
                    iterations.append(float(num_iterations))
                else:
                    previous = iterations[len(iterations) - 1] + 0
                    # iterations.append(float(num_iterations) + previous)
                    iterations.append(float(num_iterations))
        time_steps = np.linspace(0, 1, len(iterations), endpoint=True)
    norm_fun = interpolate.interp1d(time_steps, iterations)
    c = computation_parameters[index]
    file_path = "/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/res{}_{}__{}__hho{}__{}__{}__{}/output.txt".format(batch, specimen, c["mesh"], c["order"], c["algorithm"], c["jacobian"], c["acceleration"])
    with open(file_path, 'r') as file:
        iterations = []
        lines = file.readlines()
        for line in lines:
            if "+ ITERATIONS : " in line:
                num_iterations = line.split()[3]
                if len(iterations) == 0:
                    iterations.append(float(num_iterations))
                else:
                    previous = iterations[len(iterations) - 1] + 0
                    iterations.append(float(num_iterations))
                    # iterations.append(float(num_iterations) + previous)
        time_steps = np.linspace(0, 1, len(iterations), endpoint=True)
    return time_steps, [iterations[count]/norm_fun(time_steps[count]) for count in range(len(time_steps))]

def fetch_num_cell_iters(batch, specimen, index):
    c = computation_parameters[index]
    file_path = "/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/res{}_{}__{}__hho{}__{}__{}__{}/output.txt".format(batch, specimen, c["mesh"], c["order"], c["algorithm"], c["jacobian"], c["acceleration"])
    with open(file_path, 'r') as file:
        # iterations = [[]]
        iters2 = [0]
        lines = file.readlines()
        count_steps = 0
        count_iters = 0
        for line in lines:
            if "MEAN_CELLS_ITERATIONS : " in line:
                num_iterations = line.split()[7]
                iters2[count_steps] += float(num_iterations)
                count_iters += 1
                # iterations[count_steps].append(float(num_iterations))
            if "+ ITERATIONS : " in line:
                iters2[count_steps] /= float(count_iters)
                count_steps += 1
                count_iters = 0
                iters2.append(0)
                # iterations.append([])
        iters2.pop()
        time_steps = np.linspace(0, 1, len(iters2), endpoint=True)
    return time_steps, iters2

def plot_sphere(batch, indices, norm_indices, suffix = ""):
    fig, ax = plt.subplots(1, len(indices), sharey=True, figsize=(8, 4), dpi=150)
    for cnt, index_group in enumerate(indices):
        for i_cnt, i in enumerate(index_group):
            x, y = fetch_data(batch, "swelling_sphere", i, norm_indices[cnt])
            c = computation_parameters[i]
            tag_item = "" + c["algorithm"] + c["jacobian"] + c["acceleration"]
            ptr = ax[cnt]
            ptr.plot(x, y,
                # label=lab,
                label=label_tag_rule[tag_item]["label"],
                color=label_tag_rule[tag_item]["color"],
                lw = 2,
                linestyle=label_tag_rule[tag_item]["linestyle"],
                # marker=label_tag_rule[tag_item]["marker"],
                # markevery=label_tag_rule[tag_item]["markevery"],
            )
            ptr.set_title('HHO({}, {})'.format(c["order"], c["order"]))
            if cnt == 0:
                ptr.set_ylabel('Normalized number of iteration')
        ptr.set_yticks([tick_i for tick_i in range(11)], minor=False)
        ptr.grid()
        ptr.axvline(0.5, c='k', ls=':')
        ptr.text(0.55,7,'yield stress',rotation=90)
    fig.text(0.5, 0.0, 'Normalized pseudo-time step', ha='center')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    lgd = fig.axes[0].legend(handles, labels, loc='upper center', bbox_to_anchor=(1.7,-0.15), ncol = 6)
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/plot_{}{}{}.png".format("swelling_sphere", batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/paper_n/img_calcs/plot_{}{}{}.png".format("swelling_sphere", batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_notched_rod(batch, indices, norm_indices, suffix = ""):
    fig, ax = plt.subplots(1, len(indices), sharey=True, figsize=(8, 4), dpi=150)
    for cnt, index_group in enumerate(indices):
        for i_cnt, i in enumerate(index_group):
            x, y = fetch_data(batch, "notched_rod", i, norm_indices[cnt])
            c = computation_parameters[i]
            tag_item = "" + c["algorithm"] + c["jacobian"] + c["acceleration"]
            ptr = ax[cnt]
            ptr.plot(x, y,
                label=label_tag_rule[tag_item]["label"],
                color=label_tag_rule[tag_item]["color"],
                lw = 2,
                linestyle=label_tag_rule[tag_item]["linestyle"],
                # marker=label_tag_rule[tag_item]["marker"],
                # markevery=label_tag_rule[tag_item]["markevery"],
            )
            ptr.set_title('HHO({}, {})'.format(c["order"], c["order"]))
            if cnt == 0:
                ptr.set_ylabel('Normalized number of iteration')
        # ptr.set_yticks([tick_i for tick_i in range(11)], minor=False)
        ptr.grid()
    fig.text(0.5, 0.0, 'Normalized pseudo-time step', ha='center')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    lgd = fig.axes[0].legend(handles, labels, loc='upper center', bbox_to_anchor=(1.7,-0.15), ncol = 6)
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/plot_{}{}{}.png".format("notched_rod", batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/paper_n/img_calcs/plot_{}{}{}.png".format("notched_rod", batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_notched_rod_num_cell_iterations(batch, indices, suffix = ""):
    specimen = "notched_rod"
    fig, ax = plt.subplots(1, len(indices), sharey=True, figsize=(8, 4), dpi=150)
    for cnt, index_group in enumerate(indices):
        for i_cnt, i in enumerate(index_group):
            x, y = fetch_num_cell_iters(batch, specimen, i)
            c = computation_parameters[i]
            tag_item = "" + c["algorithm"] + c["jacobian"] + c["acceleration"]
            ptr = ax[cnt]
            ptr.plot(x, y,
                label=label_tag_rule[tag_item]["label"],
                color=label_tag_rule[tag_item]["color"],
                lw = 2,
                linestyle=label_tag_rule[tag_item]["linestyle"],
                # marker=label_tag_rule[tag_item]["marker"],
                # markevery=label_tag_rule[tag_item]["markevery"],
            )
            ptr.set_title('HHO({}, {})'.format(c["order"], c["order"]))
            if cnt == 0:
                ptr.set_ylabel('Mean number of local cell iterations')
        ptr.set_yticks([tick_i for tick_i in range(6)], minor=False)
        ptr.grid()
    fig.text(0.5, 0.0, 'Normalized pseudo-time step', ha='center')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    lgd = fig.axes[0].legend(handles, labels, loc='upper center', bbox_to_anchor=(1.7,-0.15), ncol = 6)
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/plot_{}{}{}.png".format(specimen, batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/paper_n/img_calcs/plot_{}{}{}.png".format(specimen, batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_swelling_sphere_num_cell_iterations(batch, indices, suffix = ""):
    specimen = "swelling_sphere"
    fig, ax = plt.subplots(1, len(indices), sharey=True, figsize=(8, 4), dpi=150)
    for cnt, index_group in enumerate(indices):
        for i_cnt, i in enumerate(index_group):
            x, y = fetch_num_cell_iters(batch, specimen, i)
            c = computation_parameters[i]
            tag_item = "" + c["algorithm"] + c["jacobian"] + c["acceleration"]
            ptr = ax[cnt]
            ptr.plot(x, y,
                label=label_tag_rule[tag_item]["label"],
                color=label_tag_rule[tag_item]["color"],
                lw = 2,
                linestyle=label_tag_rule[tag_item]["linestyle"],
                # marker=label_tag_rule[tag_item]["marker"],
                # markevery=label_tag_rule[tag_item]["markevery"],
            )
            ptr.set_title('HHO({}, {})'.format(c["order"], c["order"]))
            if cnt == 0:
                ptr.set_ylabel('Mean number of local cell iterations')
        ptr.set_yticks([tick_i for tick_i in range(11)], minor=False)
        ptr.grid()
    fig.text(0.5, 0.0, 'Normalized pseudo-time step', ha='center')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    lgd = fig.axes[0].legend(handles, labels, loc='upper center', bbox_to_anchor=(1.7,-0.15), ncol = 6)
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/plot_{}{}{}.png".format(specimen, batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/paper_n/img_calcs/plot_{}{}{}.png".format(specimen, batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_memory_footprint():

    def get_value_static(
        ord_cell, ord_face, num_faces, dim_field, dim_euclidean, accelerate = 0
    ):
        cell_size = basis_dimension = int(binom(ord_cell + dim_euclidean, ord_cell))
        face_size = basis_dimension = int(binom(ord_face + dim_euclidean, ord_face))
        bas =  dim_field * cell_size * (cell_size + face_size * num_faces)
        return bas + 2 * accelerate * dim_field * (face_size * num_faces + cell_size)

    def get_value_cell_eq(
        ord_cell, ord_face, num_faces, dim_field, dim_euclidean, accelerate = 0
    ):
        face_size = basis_dimension = int(binom(ord_face + dim_euclidean, ord_face))
        bas = 0
        return bas + 2 * accelerate * dim_field * (face_size * num_faces)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    
    for acceleration in range(0, 3):
        for method in ["Static", "Cell"]:
            count = 0
            x, y, ticks, ticks_labels = [], [], [], []
            for face_order in range(1, 4):
                for cell_order in range(face_order - 1, face_order + 2):
                    value = 0
                    if method == "Static":
                        value = get_value_static(cell_order, face_order, 4, 2, 2, acceleration)
                    else:
                        value = get_value_cell_eq(cell_order, face_order, 4, 2, 2, acceleration)
                    x.append(count)
                    y.append(value)
                    ticks.append(count)
                    ticks_labels.append("HHO({},{})".format(cell_order, face_order, count))
                    count  += 1
                    print(count, value)
            lss = ['-', '--', '-.']
            item = "{}{}".format(method, acceleration)
            rules = {
                "Static0" : r'Sc',
                "Cell0" : r'Cc',
                "Static1" : r'Sa, $N_{it} = 1$',
                "Cell1" : r'Ca, $N_{it} = 1$',
                "Static2" : r'Sa, $N_{it} = 2$',
                "Cell2" : r'Ca, $N_{it} = 2$',
            }
            if method == "Static":
                ax.plot(ticks, y, c='r', ls=lss[acceleration], label=rules[item])
            else:
                ax.plot(ticks, y, c='b', ls=lss[acceleration], label=rules[item])
            ax.set_yticks([tick_i for tick_i in range(0, 2500, 500)], minor=False)
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks_labels, rotation=45, ha='right')
            ax.set_ylabel("Number of scalar entries to store")
            ax.grid()
    # ax.set_xticklabels(ticks)
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(ticks_labels)
    # ax.xticks(rotation = 45)
    # fig.text(0.0, -0.1, 'Normalized pseudo-time step', ha='center')
    fig.axes[0].grid()
    handles, labels = fig.axes[0].get_legend_handles_labels()
    lgd = fig.axes[0].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.45,-0.25), ncol = 3)
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/plot_memory.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/paper_n/img_calcs/plot_memory.png", bbox_extra_artists=(lgd,), bbox_inches='tight')




def plot_common_cell_iters(batch, indices, suffix = ""):
    specimens = ["notched_rod", "swelling_sphere"]
    spac_lab = ["notched rod", "swelling sphere"]
    fig, ax = plt.subplots(2, len(indices), sharey=True, figsize=(8, 8), dpi=150)
    for sp_c, specimen in enumerate(specimens):
        for cnt, index_group in enumerate(indices):
            for i_cnt, i in enumerate(index_group):
                x, y = fetch_num_cell_iters(batch, specimen, i)
                c = computation_parameters[i]
                tag_item = "" + c["algorithm"] + c["jacobian"] + c["acceleration"]
                ptr = ax[sp_c, cnt]
                ptr.plot(x, y,
                    label=label_tag_rule[tag_item]["label"],
                    color=label_tag_rule[tag_item]["color"],
                    lw = 2,
                    linestyle=label_tag_rule[tag_item]["linestyle"],
                    # marker=label_tag_rule[tag_item]["marker"],
                    # markevery=label_tag_rule[tag_item]["markevery"],
                )
                if specimen == "swelling_sphere":
                    ptr.axvline(0.5, c='k', ls=':')
                    ptr.text(0.55,5,'yield stress',rotation=90)
                ptr.set_title('HHO({}, {})'.format(c["order"], c["order"]))
                if cnt == 0:
                    # ptr.set_ylabel("Normalized number of iteration\nfor the {} test case".format(spac_lab[sp_c]))
                    ptr.set_ylabel("Mean number of local cell iterations\nfor the {} test case".format(spac_lab[sp_c]))
            # ptr.set_yticks([tick_i for tick_i in range(11)], minor=False)
            ptr.grid()
    fig.text(0.5, 0.055, 'Normalized pseudo-time step', ha='center')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    lgd = fig.axes[0].legend(handles, labels, loc='upper center', bbox_to_anchor=(1.55,-1.4), ncol = 6)
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/plot_cell_iterations_{}{}.png".format(batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/paper_n/img_calcs/plot_cell_iterations_{}{}.png".format(batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_common(batch, indices, norm_indices, suffix = ""):
    specimens = ["notched_rod", "swelling_sphere"]
    spac_lab = ["notched rod", "swelling sphere"]
    fig, ax = plt.subplots(2, len(indices), sharey=True, figsize=(8, 8), dpi=150)
    for sp_c, specimen in enumerate(specimens):
        for cnt, index_group in enumerate(indices):
            for i_cnt, i in enumerate(index_group):
                x, y = fetch_data(batch, specimen, i, norm_indices[cnt])
                c = computation_parameters[i]
                tag_item = "" + c["algorithm"] + c["jacobian"] + c["acceleration"]
                ptr = ax[sp_c, cnt]
                ptr.plot(x, y,
                    label=label_tag_rule[tag_item]["label"],
                    color=label_tag_rule[tag_item]["color"],
                    lw = 2,
                    linestyle=label_tag_rule[tag_item]["linestyle"],
                    # marker=label_tag_rule[tag_item]["marker"],
                    # markevery=label_tag_rule[tag_item]["markevery"],
                )
                if specimen == "swelling_sphere":
                    ptr.axvline(0.5, c='k', ls=':')
                    ptr.text(0.55,9,'yield stress',rotation=90)
                ptr.set_title('HHO({}, {})'.format(c["order"], c["order"]))
                if cnt == 0:
                    ptr.set_ylabel("Normalized number of iteration\nfor the {} test case".format(spac_lab[sp_c]))
            # ptr.set_yticks([tick_i for tick_i in range(11)], minor=False)
            ptr.grid()
    fig.text(0.5, 0.055, 'Normalized pseudo-time step', ha='center')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    lgd = fig.axes[0].legend(handles, labels, loc='upper center', bbox_to_anchor=(1.55,-1.4), ncol = 6)
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/plot_global_iterations_{}{}.png".format(batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/paper_n/img_calcs/plot_global_iterations_{}{}.png".format(batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')


plot_memory_footprint() 


plot_sphere(
    "_4",
    [
        [2, 3, 1, 4, 5], [8, 9, 7, 10, 11], [14, 15, 13, 16, 17]
    ],
    [0, 6, 12],
    "_ordn"
)
plot_notched_rod(
    "_4",
    [
        [2, 3, 1, 4, 5], [8, 9, 7, 10, 11], [14, 15, 13, 16, 17]
    ],
    [0, 6, 12],
    "_ordn"
)
plot_notched_rod(
    "_6",
    [
        [2, 3, 1, 4, 5], [2, 3, 1, 4, 5]
    ],
    [0, 0],
    "_ordn"
)

ts, its = fetch_num_cell_iters(
    "_4",
    "swelling_sphere",
    11
)

plot_notched_rod_num_cell_iterations(
    "_4",
    [
        [1, 4, 5], [7, 10, 11], [13, 16, 17]
    ],
    "_cell_iters"
)

plot_swelling_sphere_num_cell_iterations(
    "_4",
    [
        [1, 4, 5], [7, 10, 11], [13, 16, 17]
    ],
    "_cell_iters"
)

plot_common(
    "_4",
    [
        [2, 3, 1, 4, 5], [8, 9, 7, 10, 11], [14, 15, 13, 16, 17]
    ],
    [0, 6, 12],
    "_ordn"
)

plot_common_cell_iters(
    "_4",
    [
        [1, 4, 5], [7, 10, 11], [13, 16, 17]
    ],
    "_cell_iters"
)

plt.close()
plt.plot(ts, its)
plt.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/plot_num_iter.png")
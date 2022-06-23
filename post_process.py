import os
from mgis import behaviour as mgis_bv 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 12})

# rc('font', **{'family':'serif', 'serif':['Computer Modern Roman']})
rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 
                                'monospace': ['Computer Modern Typewriter']})
params = {'backend': 'pdf',
            'font.size' : 16,
          'axes.labelsize': 16,
          'legend.fontsize': 16,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16,
          'text.usetex': True,
          'axes.unicode_minus': True}
matplotlib.rcParams.update(params)

# batch = "_2"
# batch = ""

computation_parameters = [
    # --- ORDER 1
    # CONSISTENT JACOBIAN
    {"mesh" : "coarse", "order": 1, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 1, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator, "acceleration": 0},
    # ELASTIC JACOBIAN
    {"mesh" : "coarse", "order": 1, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 1, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 1},
    {"mesh" : "coarse", "order": 1, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 2},
    {"mesh" : "coarse", "order": 1, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 1, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 1},
    # --- ORDER 2
    # CONSISTENT JACOBIAN
    {"mesh" : "coarse", "order": 2, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 2, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator, "acceleration": 0},
    # ELASTIC JACOBIAN
    {"mesh" : "coarse", "order": 2, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 2, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 1},
    {"mesh" : "coarse", "order": 2, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 2},
    {"mesh" : "coarse", "order": 2, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 2, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 1},
    # --- ORDER 3
    # CONSISTENT JACOBIAN
    {"mesh" : "coarse", "order": 3, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 3, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator, "acceleration": 0},
    # ELASTIC JACOBIAN
    {"mesh" : "coarse", "order": 3, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 3, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 1},
    {"mesh" : "coarse", "order": 3, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 2},
    {"mesh" : "coarse", "order": 3, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 3, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 1},
]

algo_dict = {
    "explicit" : {"label": "Static Condensation", "linestyle": "--", "color": 'b'},
    "implicit" : {"label": "Cell Equilibrium", "linestyle": ":", "color": 'r'},
}

integration_dict = {
    mgis_bv.IntegrationType.IntegrationWithElasticOperator : {"label": "elastic jacobian", "linestyle": '--', "marker": 'p'},
    mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator : {"label": "consistent jacobian", "linestyle": '-', "marker": ''}
}

acceleration_dict = {
    0 : {"label": " no acceleration", "linestyle" : ':'},
    1 : {"label": "global acceleration", "linestyle" : '--'},
    2 : {"label": "separate acceleration", "linestyle" : '-.'}
}

# linestyle_dict = {
#     "explicit" : '--',
#     "implicit" : '-.',
# }

def fetch_data(batch, specimen, index, norm_index):
    c_norm = computation_parameters[norm_index]
    file_path = "/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/res{}_{}__{}__hho{}__{}__{}__acceleration_{}/output.txt".format(batch, specimen, c_norm["mesh"], c_norm["order"], c_norm["algorithm"], c_norm["integration"], c_norm["acceleration"])
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
    file_path = "/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/res{}_{}__{}__hho{}__{}__{}__acceleration_{}/output.txt".format(batch, specimen, c["mesh"], c["order"], c["algorithm"], c["integration"], c["acceleration"])
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


def plot(batch, specimen, indices, norm_indices, suffix = ""):
    colors = ['r', 'g', 'b', 'm', 'c', 'k', 'y']
    # fig = plt.figure(figsize=(8, 6.5), dpi=150)
    fig, ax = plt.subplots(1, len(indices), sharey=True, figsize=(8, 6.5), dpi=150)
    for cnt, index_group in enumerate(indices):
        for i_cnt, i in enumerate(index_group):
            x, y = fetch_data(batch, specimen, i, norm_indices[cnt])
            c = computation_parameters[i]
            # lab = "{}, HHO({},{}), {}, {}".format(algo_dict[c["algorithm"]], c["order"], c["order"], integration_dict[c["integration"]], acceleration_dict[c["acceleration"]])
            lab = "{}, {}, {}".format(algo_dict[c["algorithm"]]["label"], integration_dict[c["integration"]]["label"], acceleration_dict[c["acceleration"]]["label"])
            # for kk in range(len(time_steps)):
            #     print(iterations[kk]/norm_fun(time_steps[kk]))
            ptr = ax[cnt]
            ptr.plot(x, y,
                label=lab,
                # color=algo_dict[c["algorithm"]]["color"],
                color=colors[i_cnt],
                # marker=integration_dict[c["integration"]]["marker"],
                lw = 2,
                linestyle=acceleration_dict[c["acceleration"]]["linestyle"],
                # markevery=5
            )
            ptr.set_title('HHO({}, {})'.format(c["order"], c["order"]))
            # ptr.set_xlabel('normalized pseudo-time step')
            if cnt == 0:
                ptr.set_ylabel('normalized number of iteration')
        # ptr.set_yticks([tick_i for tick_i in range(11)], minor=False)
        ptr.grid()
        # if specimen == "swelling_sphere":
            # ax.set_ylim(0, 10)
            # ptr.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=1)
            # ptr.legend()
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.9])
        # ax.figure.set_size_inches(8, 9)
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=1)
    # ax.legend()
    # fig.set_xlabel('normalized pseudo-time step')
    # fig.set_ylabel('normalized number of iteration')
    fig.text(0.5, 0.02, 'normalized pseudo-time step', ha='center')
    # fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
    # fig.supxlabel('normalized pseudo-time step')
    # fig.supylabel('normalized number of iteration')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    lgd = fig.axes[0].legend(handles, labels, loc='upper center', bbox_to_anchor=(1.7,-0.15))
    # text = fig.axes[-1].text(-0.2,1.05, "", transform=fig.axes[-1].transAxes)
    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc = 'lower center', bbox_to_anchor=(0.5,-0.1))
    # fig.legend()
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/plot_{}{}{}.png".format(specimen, batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')
    # fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/h2o/plot_{}{}{}.png".format(specimen, batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig("/home/dsiedel/Documents/2022_01_06_PAPER_01/paper_n/img_calcs/plot_{}{}{}.png".format(specimen, batch, suffix), bbox_extra_artists=(lgd,), bbox_inches='tight')

# plot("", "swelling_sphere", [
#     0, 1, 2, 3, 4, 5
# ])
# plot("", "notched_rod", [
#     # 0,
#     1, 2, 3, 4, 5])
# plot("_2", "swelling_sphere", [
#     # 0,
#     1, 2, 4, 5, 12
# ], "_ord1")
plot("_3", "swelling_sphere", [
    # [1, 2, 3, 4, 5, 6], [8, 9 ,10 ,11, 12, 13], [15, 16 ,17 ,18, 19, 20]
    # [1, 2, 4, 5, 6], [8, 9 ,11, 12, 13], [15, 16 ,18, 19, 20]
    [1, 2, 5, 6], [8, 9, 12, 13], [15, 16, 19, 20]
    # [8, 9, 10, 11, 12]
], [0, 7, 14], "_ordn")
plot("_3", "notched_rod", [
    # [1, 2, 3, 4, 5, 6], [8, 9 ,10 ,11, 12, 13], [15, 16 ,17 ,18, 19, 20]
    # [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]
    [1, 2, 5, 6], [1, 2, 5, 6], [1, 2, 5, 6]
    # [8, 9, 10, 11, 12]
], [0, 0, 0], "_ordn")


# fig, ax = plt.subplots(2, 2)
# for i in range(2):
#     ax[1][i].plot(x, np.tan(x+i),
#                  label = "y=tan(x+{})".format(i))
      
#     ax[0][i].plot(x, np.tan(x+i), 
#                   label = "y=tan(x+{})".format(i))
      
# # Add legends

# fig.legend(bbox_to_anchor=(1.3, 0.6))
  
# # Show plot

# fig.tight_layout() 
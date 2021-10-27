import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)
from matplotlib.colors import ListedColormap
import copy
import MDPSolver as MDPSolver
import os
import matplotlib as mpl
mpl.rcParams['font.serif'] = ['times new roman']
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
mpl.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 32})
mpl.rc('legend', **{'fontsize': 22})
mpl.rc('text', usetex=True)
# fig_size = [6.4, 4.8]


import matplotlib as mpl
# mpl.rcParams['font.serif'] = ['times new roman']
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
# plt.rcParams["figure.figsize"] = [3, 3]
# mpl.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 25})
# mpl.rc('legend', **{'fontsize': 11})
# mpl.rc('text', usetex=True)


myCmap = 'RdBu'

def input_from_file(dir_path, filename):
    file_path = "{}/{}".format(dir_path, filename)
    dict_file = {}
    t=int(1e6)

    int_keys = ['n_states', 'n_actions', 'gridsizefull']
    with open(file_path) as f:
        for line in f:
            read_line = line.split()
            if read_line[0] in int_keys:
                dict_file[read_line[0]] = int(read_line[1])
            elif read_line[0] == "H_set":
                dict_file[read_line[0]] = np.array(list(map(int, read_line[1:t])))
            else:
                dict_file[read_line[0]] = np.array(list(map(float, read_line[1:t])))
    return dict_file
#enddef


# exit(0)

def mapping_rewards_rom_abstraction_to_orig_env(env_orig, env_abstr):

    reward = np.zeros((201, 3))
    env_with_anstracted_reward = copy.deepcopy(env_orig)
    for s_i, s_orig in enumerate(env_orig.states_in_zero_one_interval_with_key):
        s_abtracted_int = env_abstr.mapping_from_con_state_to_int_state(s_orig)
        for a in range(env_orig.n_actions):
            reward[s_i, a] = env_abstr.reward[s_abtracted_int, a]
    return reward
#enddef

def get_env_with_new_R(env_orig, reward):
    env_new = copy.deepcopy(env_orig)
    env_new.reward = copy.deepcopy(reward)
    return env_new
#ednndef

def highlight_cell_vert(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 0,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def highlight_cell_hor(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,0, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def plot_reward_action(reward, name, color_ranges):
    plt.rcParams["figure.figsize"] = [14.5, 3.5]

    fig, (ax1) = plt.subplots(nrows=1, ncols=5, sharex=True)
    # fig.suptitle("Without key")

    ####### plot R_S###########
    myCmap_reward = ListedColormap(['white', '#808080'])

    reward_to_plot = np.sum(abs(reward), axis=1)

    ax1[0].imshow(np.rot90(reward_to_plot[:-1].reshape(7, 7)),
               cmap=myCmap_reward, aspect="auto", vmin=0, vmax=0.1)
    ax1[0].set_yticks([])
    ax1[0].set_xlabel(r"$R(s, \cdot)\neq0$", fontsize=26)

    ### vertical walls
    highlight_cell_vert(3, 0, ax=ax1[0], color="#B54407", linewidth=5)
    highlight_cell_vert(3, 2, ax=ax1[0],  color="#B54407", linewidth=5)
    highlight_cell_vert(3, 3,ax=ax1[0],  color="#B54407", linewidth=5)
    highlight_cell_vert(3, 4,ax=ax1[0],  color="#B54407", linewidth=5)
    # highlight_cell(3, 5, color="#B54407", linewidth=5)
    highlight_cell_vert(3, 6,ax=ax1[0],  color="#B54407", linewidth=5)

    #horizontal walls

    highlight_cell_hor(0, 4,ax=ax1[0],  color="#B54407", linewidth=5)
    highlight_cell_hor(2, 4,ax=ax1[0],  color="#B54407", linewidth=5)
    highlight_cell_hor(3, 4,ax=ax1[0],  color="#B54407", linewidth=5)
    highlight_cell_hor(4, 4,ax=ax1[0],  color="#B54407", linewidth=5)
    # highlight_cell(3, 5, color="#B54407", linewidth=5)
    highlight_cell_hor(6, 4,ax=ax1[0],  color="#B54407", linewidth=5)

    #UP action
    UP_action_rewarrd_to_plot = np.flip(reward[:, 0][:-1].reshape(7, 7), axis=0)
    ax1[1].imshow(UP_action_rewarrd_to_plot,
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)

    # text portion
    ind_array = np.arange(0, 7, 1)
    x, y = np.meshgrid(ind_array, ind_array)
    # x = np.rot90(x)
    # y = np.rot90(y)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if UP_action_rewarrd_to_plot[int(x_val), int(y_val)] <= 0 - 1e-7:
            c=r"$-$"
        elif UP_action_rewarrd_to_plot[int(x_val), int(y_val)] >= 0+1e-7:
            c=r"$+$"
        else:
            c=""
        ax1[1].text(y_val, x_val, c, va='center', ha='center', fontsize=22)


    ax1[1].set_yticks([])
    ax1[1].set_xlabel(r'$R(s,\textnormal{``up"})$', fontsize=26)



    #LEFT action
    LEFT_action_rewarrd_to_plot = np.flip(reward[:, 1][:-1].reshape(7, 7), axis=0)

    # fig, (ax) = plt.subplots(nrows=1, sharex=True)
    ax1[2].imshow(LEFT_action_rewarrd_to_plot,
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    # text portion
    ind_array = np.arange(0, 7, 1)
    x, y = np.meshgrid(ind_array, ind_array)
    # x = np.rot90(x)
    # y = np.rot90(y)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if LEFT_action_rewarrd_to_plot[int(x_val), int(y_val)] <= 0 - 1e-7:
            c = r"$-$"
        elif LEFT_action_rewarrd_to_plot[int(x_val), int(y_val)] >= 0 + 1e-7:
            c = r"$+$"
        else:
            c = ""
        ax1[2].text(y_val, x_val, c, va='center', ha='center', fontsize=22)

    ax1[2].set_yticks([])
    ax1[2].set_xticks([])
    ax1[2].set_xlabel(r'$R(s,\textnormal{``left"})$', fontsize=26)

    #DOWN action
    DOWN_action_rewarrd_to_plot = np.flip(reward[:, 2][:-1].reshape(7, 7), axis=0)
    ax1[3].imshow(DOWN_action_rewarrd_to_plot,
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    # text portion
    ind_array = np.arange(0, 7, 1)
    x, y = np.meshgrid(ind_array, ind_array)
    # x = np.rot90(x)
    # y = np.rot90(y)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if DOWN_action_rewarrd_to_plot[int(x_val), int(y_val)] <= 0 - 1e-7:
            c = r"$-$"
        elif DOWN_action_rewarrd_to_plot[int(x_val), int(y_val)] >= 0 + 1e-7:
            c = r"$+$"
        else:
            c = ""
        ax1[3].text(y_val, x_val, c, va='center', ha='center', fontsize=22)
    ax1[3].set_yticks([])
    ax1[3].set_xlabel(r'$R(s,\textnormal{``down"})$', fontsize=26)

    # RIGHT action

    RIGHT_action_rewarrd_to_plot = np.flip(reward[:, 3][:-1].reshape(7, 7), axis=0)
    ax1[4].imshow(RIGHT_action_rewarrd_to_plot,
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    # text portion
    ind_array = np.arange(0, 7, 1)
    x, y = np.meshgrid(ind_array, ind_array)
    # x = np.rot90(x)
    # y = np.rot90(y)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if RIGHT_action_rewarrd_to_plot[int(x_val), int(y_val)] <= 0 - 1e-7:
            c = r"$-$"
        elif RIGHT_action_rewarrd_to_plot[int(x_val), int(y_val)] >= 0 + 1e-7:
            c = r"$+$"
        else:
            c = ""
        ax1[4].text(y_val, x_val, c, va='center', ha='center', fontsize=22)
    ax1[4].set_yticks([])
    ax1[4].set_xlabel(r'$R(s,\textnormal{``right"})$', fontsize=26)

    # plt.xticks([0, 6], ["1","7"], fontsize=13)
    # plt.show()
    plt.tight_layout()
    # plt.ylabel(, fontsize=16)


    plt.tight_layout()
    # plt.ylabel(, fontsize=16)
    plt.savefig(name, bbox_inches='tight')



    pass



def calculate_PBRS_orig(env_orig, env_abstr, n_times_state_piced=10,
                        n_times_action_picked=10):

    _, V_abstract, _, _ = MDPSolver.valueIteration(env_abstr, env_abstr.reward)

    reward_PBRS_int = np.zeros((len(env_orig.states_in_zero_one_interval_with_key), 3))

    T, R = discretization_chain_0_1_new_key.get_sampling_T_ground(env_orig, n_times_state_piced, n_times_action_picked)
    reward_PBRS = {}
    for state in env_orig.states_in_zero_one_interval_with_key:
        for action in range(env_orig.n_actions):
            sum_over_next_state = 0
            for next_state in env_orig.states_in_zero_one_interval_with_key:
                next_state = tuple(next_state)
                sum_over_next_state += env_orig.gamma * T[state, next_state, action] * \
                    V_abstract[env_abstr.mapping_from_con_state_to_int_state(next_state)]
            reward_PBRS[state, action] = R[state, action] + sum_over_next_state - \
                    V_abstract[env_abstr.mapping_from_con_state_to_int_state(state)]
            reward_PBRS_int[env_orig.mapping_from_con_state_to_int_state(state), action] = reward_PBRS[state, action]




    return reward_PBRS, reward_PBRS_int
#enddef

def map_PBRS_to_orig_environment(env_orig, R_PBRS_int):
    reward = {}
    env_orig_tmp = copy.deepcopy(env_orig)
    for i, s in enumerate(env_orig.states_in_zero_one_interval_with_key):
        for a in range(env_orig.n_actions):
            reward[s, a] = R_PBRS_int[i, a]
    env_orig_tmp.reward = reward
    return env_orig_tmp
#enddef


def plotting(dict_file, folder_name, color_ranges=None):
    directory = "output/"+folder_name+"_color_ranges={}".format(color_ranges)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    R_dsgn = dict_file["R_reward_design_env=room_n_states=50_len_active_set_chosen=6"].reshape(50, 4)
    #### DESIGN#####

    plot_reward_action(R_dsgn,
                       directory+"/fourroom_visual_exprd.pdf", color_ranges=color_ranges)

    pass
#endef

def plotting(dict_file, color_ranges, B=None, name="exprd"):

    directory = "output".format(color_ranges)
    if not os.path.isdir(directory):
        os.makedirs(directory)


    if name == "exprd":
        R_dsgn = dict_file["R_reward_design_env=room_n_states=50_len_active_set_chosen={}".format(B+1)].reshape(50, 4)

        plot_reward_action(R_dsgn,
                           directory + "/exprd_B={}_lambda=0.pdf".format(B),
                           color_ranges)
    elif name == "pbrs":
        R_pot = dict_file["R_potential_shaping"].reshape(50, 4)

        plot_reward_action(R_pot,
                           directory + "/pbrs.pdf", color_ranges)
    elif name == "orig":
        R_orig = dict_file["R_original"].reshape(50, 4)

        plot_reward_action(R_orig,
                           directory + "/orig.pdf",
                           color_ranges)
    else:
        print("Algorithm names should be chosen from {orig, pbrs, exprd}")
        exit(0)
    pass
#enddef

if __name__ == "__main__":
    filename = "information_is_delta_s_const_{}_1.txt".format(False)
    dir_path = "results/designed_rewards_room_const_middle"

    tol = 1e-10
    dict_file = input_from_file(dir_path, filename)


    color_ranges = 4

    plotting(dict_file, color_ranges)

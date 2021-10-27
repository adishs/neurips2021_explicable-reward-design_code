import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)
plt.rcParams["figure.figsize"] = [3.3, 3.8]
from matplotlib.colors import ListedColormap
import copy
import MDPSolver
import os




import matplotlib as mpl
# mpl.rcParams['font.serif'] = ['times new roman']
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
# plt.rcParams["figure.figsize"] = [3, 3]
# mpl.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 25})
# mpl.rc('legend', **{'fontsize': 11})
# mpl.rc('text', usetex=True)


myCmap = ListedColormap(['red', 'white', 'blue'])

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

def plot_reward_action(reward, name, color_ranges):

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, sharex=True)
    # fig.suptitle("Without key")

    #UP action

    ax1[0].imshow(np.rot90(reward[:, 0][:-1].reshape(7, 7)),
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax1[0].set_yticks([])
    ax1[0].set_xlabel("up", fontsize=15)

    #LEFT action

    # fig, (ax) = plt.subplots(nrows=1, sharex=True)
    ax1[1].imshow(np.rot90(reward[:, 1][:-1].reshape(7, 7)),
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax1[1].set_yticks([])
    ax1[1].set_xticks([])
    ax1[1].set_xlabel("left", fontsize=15)

    #DOWN action
    ax2[0].imshow(np.rot90(reward[:, 2][:-1].reshape(7, 7)),
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax2[0].set_yticks([])
    ax2[0].set_xlabel("down", fontsize=15)

    # RIGHT action

    ax2[1].imshow(np.rot90(reward[:, 3][:-1].reshape(7, 7)),
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax2[1].set_yticks([])
    ax2[1].set_xlabel("right", fontsize=15)

    # plt.xticks([0, 6], ["1","7"], fontsize=13)
    # plt.show()
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
    directory = "results/"+folder_name+"_color_ranges={}".format(color_ranges)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    R_dsgn = dict_file["R_reward_design_env=room_n_states=50_len_active_set_chosen=6"].reshape(50, 4)
    #### DESIGN#####

    plot_reward_action(R_dsgn,
                       directory+"/fourroom_visual_exprd.pdf", color_ranges=color_ranges)

    pass
#endef



if __name__ == "__main__":
    pass

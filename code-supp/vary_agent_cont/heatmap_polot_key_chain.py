import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)
plt.rcParams["figure.figsize"] = [3.5, 4]
from matplotlib.colors import ListedColormap
import copy
import linekey_MDPSolver as MDPSolver
import os

import linekey_env_cont
import linekey_abstraction

def get_potential_based_reward(env, tol=1e-10):
    R_pot = copy.deepcopy(env.reward)
    Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, env.reward, tol=tol)
    V = np.round(V, 10)
    for s in range(env.n_states):
        for a in range(env.n_actions):
            sum_over_next_states = 0
            for n_s in range(env.n_states):
                sum_over_next_states += env.gamma * env.T[s, n_s, a] * V[n_s]
            R_pot[s, a] += sum_over_next_states - V[s]

    return R_pot
#enddef

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

    reward = np.zeros((401, 3))
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
    rect = plt.Rectangle((x-.5, y-.5), 1,0, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 20,20, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def highlight_cell_hor(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 20,20, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def highlight_cell_hor_1(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 10,9, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def plot_reward_action(reward_orig_abstracted_reward, name, color_ranges):

    fig, (ax1,ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(nrows=8, sharex=True)
    # fig.suptitle("Without key")


    ####### plot R_S###########
    myCmap_reward = ListedColormap(['white', '#808080'])

    reward_to_plot = np.sum(abs(reward_orig_abstracted_reward), axis=1)

    ax1.imshow([reward_to_plot[:200]],
                  cmap=myCmap_reward, aspect="auto", vmin=0, vmax=0.1)
    ax1.set_yticks([])
    ax1.set_ylabel("-", fontsize=16)
    # ax7.set_xlabel(r"$R(s, \cdot)\neq0$", fontsize=26)


    ax2.imshow([reward_to_plot[200:400]],
                  cmap=myCmap_reward, aspect="auto", vmin=0, vmax=0.1)
    ax2.set_yticks([])
    # ax8.set_xlabel(r"$R(s, \cdot)\neq0$", fontsize=26)
    ax2.set_ylabel("K", fontsize=16)

    highlight_cell_vert(20, 1, ax=ax1, color="#AFEEEE", linewidth=35)
    highlight_cell_vert(40, 1, ax=ax1, color="#AFEEEE", linewidth=35)
    highlight_cell_hor(20, 1, ax=ax1, color="#AFEEEE", linewidth=4)
    highlight_cell_hor(20, 0, ax=ax1, color="#AFEEEE", linewidth=4)


    # highlight_cell_vert(90, 1, ax=ax2, color="#00FFFF", linewidth=10)
    # highlight_cell_vert(100, 1, ax=ax2, color="#00FFFF", linewidth=40)
    highlight_cell_hor(180, 1, ax=ax2, color="#00b300", linewidth=4)
    highlight_cell_hor(180, 0, ax=ax2, color="#00b300", linewidth=4)


    ax3.imshow([reward_orig_abstracted_reward[:, 0][:200]],
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax3.set_yticks([])
    ax3.set_ylabel("l-", fontsize=16)
    # ax1.ylabel('ylabel', fontsize=16)

    # ax1.set_title('Action Left')

    # fig, (ax) = plt.subplots(nrows=1, sharex=True)
    ax4.imshow([reward_orig_abstracted_reward[:, 1][:200]],
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax4.set_yticks([])
    ax4.set_ylabel("r-", fontsize=16)
    # ax2.ylabel('ylabel', fontsize=16)
    # ax2.set_title('Action Right')


    # fig, (ax) = plt.subplots(nrows=1, sharex=True)
    ax5.imshow([reward_orig_abstracted_reward[:, 2][:200]],
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax5.set_yticks([])
    ax5.set_xlim(0, 199)
    ax5.set_ylabel("p-", fontsize=16)
    # ax3.ylabel('ylabel', fontsize=16)
    # ax3.set_title('Action Pick')

    # plt.tight_layout()
    # plt.savefig("fig_without_key.pdf", bbox_inches='tight')


    # fig, (ax1,ax2, ax3) = plt.subplots(nrows=6, sharex=True)

    ax6.imshow([reward_orig_abstracted_reward[:, 1][200:400]],
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax6.set_yticks([])
    # ax4.set_title('Action Left')
    ax6.set_ylabel("rK", fontsize=16)
    # ax4.ylabel('ylabel', fontsize=16)

    # fig, (ax) = plt.subplots(nrows=1, sharex=True)
    ax7.imshow([reward_orig_abstracted_reward[:, 0][200:400]], cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax7.set_yticks([])
    # ax5.set_title('Action Right')
    ax7.set_ylabel("lK", fontsize=16)
    # ax5.ylabel('ylabel', fontsize=16)


    # fig, (ax) = plt.subplots(nrows=1, sharex=True)
    ax8.imshow([reward_orig_abstracted_reward[:, 2][200:400]],
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax8.set_yticks([])
    ax8.set_xlim(0, 199)
    ax8.set_ylabel("pK", fontsize=16)

    plt.tight_layout()
    # plt.ylabel(, fontsize=16)
    plt.savefig(name, bbox_inches='tight')

    # ax6.set_title('Action Pick')
    # ax2.set_xticks()
    plt.xticks([0, 49, 99, 149, 199], [0.0, 0.25, 0.5, 0.75, 1.0], fontsize=13)
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

def plotting(env_0_1, env_net, dict_file, color_ranges):
    R_dsgn = dict_file["R_reward_design_env=chain_n_states=41_len_active_set_chosen=7"].reshape(41, 3)
    R_pot = dict_file["R_potential_shaping"].reshape(41, 3)
    R_pot_cont = dict_file["R_potential_shaping_cont"].reshape(401, 3)
    R_dsgn_all = dict_file["Reward_all_state_design"].reshape(41, 3)
    R_orig = dict_file["R_original"].reshape(41, 3)
    directory = "results/_plots_chain_key_color_ranges_{}_SUB_IR".format(color_ranges)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    #### DESIGN ######

    env_R_dsgn = get_env_with_new_R(env_net, R_dsgn)
    reward_orig_abstracted_reward_dsgn = mapping_rewards_rom_abstraction_to_orig_env(env_0_1,
                                                                                     env_R_dsgn)

    plot_reward_action(reward_orig_abstracted_reward_dsgn,
                       directory+"/appendix_chainkey_visual_exprd_B=5_lambda=0.pdf",
                       color_ranges)

    ######## POtential ##############

    env_R_pot = get_env_with_new_R(env_net, R_pot)
    reward_orig_abstracted_reward_pot = mapping_rewards_rom_abstraction_to_orig_env(env_0_1,
                                                                                    env_R_pot)

    plot_reward_action(reward_orig_abstracted_reward_pot,
                       directory+"/appendix_chainkey_visual_pbrs_abstract.pdf", color_ranges)

    # _, reward_PBRS = calculate_PBRS_orig(env_0_1, env_net, 5, 15)
    ######## POtential CONT ##############

    plot_reward_action(R_pot_cont,
                       directory+"/appendix_chainkey_visual_pbrs_cont.pdf", color_ranges)

    ######## DESIGN_ALLL ##############

    env_R_all = get_env_with_new_R(env_net, R_dsgn_all)
    reward_orig_abstracted_reward_all = mapping_rewards_rom_abstraction_to_orig_env(env_0_1,
                                                                                    env_R_all)

    plot_reward_action(reward_orig_abstracted_reward_all,
                       directory+"/appendix_chainkey_visual_exprd_all_B=Xphi_lambda=0.pdf", color_ranges)

    ######## ORIGINAL ##############

    env_R_orig = get_env_with_new_R(env_net, R_orig)
    reward_orig_abstracted_reward_orig = mapping_rewards_rom_abstraction_to_orig_env(env_0_1,
                                                                                     env_R_orig)

    plot_reward_action(reward_orig_abstracted_reward_orig,
                       directory+"/appendix_chainkey_visual_orig.pdf",
                       color_ranges)


pass


if __name__ == "__main__":
    filename = "information_is_delta_s_const_{}_1.txt".format(False)
    dir_path = "results/designed_rewards_chain_key_const_middle"

    tol = 1e-10
    dict_file = input_from_file(dir_path, filename)


    color_ranges = 3

    env_args = {
        "R_max": 10,
        "gamma": 0.95,
        "terminalState": 1,
        "randomMoveProb": 0.1,
        "epsilon_net": 0.005,
        "n_actions": 3,
    }
    epsilon_net = 1 / 20

    env_0_1 = linekey_env_cont.discreteEnv(env_args)

    env_net = linekey_abstraction.discreteEnv(env_0_1, epsilon_net=epsilon_net,
                                                    num_traj=10, len_traj=500)

    plotting(env_0_1, env_net, dict_file, color_ranges)

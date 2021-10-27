import numpy as np
import copy
import sys
import itertools as it
import os
sys.path.append('../')
sys.path.append('../code-attacker/src')
import MDPSolver
import fourroom_env as room

import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib as mpl
mpl.rcParams['font.serif'] = ['times new roman']
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']

color_codes = ['r', 'g', 'b', '#F08080', "#8B0000", '#E802FB', "#C64C23",
 "#223206", "#7E3391", "#040004"]
# color_codes = [ 'r', 'g', 'b', '#F08080', "#8B0000", '#E802FB']
color_for_orig = "#8b0000"
color_for_pot = "#F08080"

mpl.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 32})
mpl.rc('legend', **{'fontsize': 22})
mpl.rc('text', usetex=True)
fig_size = [7, 4.8]



accumulator_for_file = {}

def write_into_file(accumulator, exp_iter, out_folder_name="out_folder", out_file_name="out_file"):
    directory = 'results/{}'.format(out_folder_name)
    filename = out_file_name + '_' + str(exp_iter) + '.txt'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filepath = directory + '/' + filename
    print("output file name  ", filepath)
    f = open(filepath, 'w')
    for key in accumulator:
        f.write(key + '\t')
        temp = list(map(str, accumulator[key]))
        for j in temp:
            f.write(j + '\t')
        f.write('\n')
    f.close()
#enddef

def write_R_into_file(env, R, s_active_set_chosen,  exp_iter,
                      env_name, exp_name="exp_name", out_file_name="output"):
    n_states = env.n_states
    directory = 'results/{}'.format(out_file_name)
    filename = 'R_{}_env={}_n_states={}_s_active_set_chosen={}_iter={}.txt'.format(exp_name, env_name, n_states,
                                                 "_".join(str(i) for i in (s_active_set_chosen)), exp_iter)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filepath = directory + '/' + filename
    np.savetxt(filepath, R)
    return
#enddef

def get_delta_s_given_policy(env_given, pi_target_d, pi_target_s, tol):

    Q_pi, _ = MDPSolver.compute_Q_V_Function_given_policy(env_given, pi_target_d,
                                                         env_given.reward, tol=tol)
    n_states = env_given.n_states
    delta_s_array = []
    for s in range(n_states):
        s_a_array = []
        for a in range(env_given.n_actions):
            if pi_target_s[s, a] == 0:
                s_a_array.append(Q_pi[s, pi_target_d[s]] - Q_pi[s, a])
        if len(s_a_array) == 0:
            s_a_array.append(0)
        delta_s_array.append(min(s_a_array))
    return delta_s_array
#enddef



def get_new_M(env_orig, new_reward):
    M = (env_orig.n_states, env_orig.n_actions, new_reward, env_orig.T, env_orig.gamma, env_orig.terminal_state)
    return M
#enddef

def calculate_I_pi_star(env, pi_star):
    n_states = env.n_states
    n_actions = env.n_actions
    I_pi_star = np.zeros((n_states * n_actions, n_states * n_actions))

    for s in range(n_states):
        opt_act = pi_star[s]
        I_pi_star[s * n_actions:s * n_actions + n_actions, s * n_actions + opt_act] = 1
        # print(s*n_actions,":",s*n_actions+n_actions)
        # print(s*n_actions + opt_act)
        # input()
    return I_pi_star
# enddef

def calculate_P_pi_star(env, pi_star):
    n_states = env.n_states
    n_actions = env.n_actions
    P_0 = env.T

    P_pi_star = np.zeros((n_states * n_actions, n_states * n_actions))

    for s in range(n_states):
        for a in range(n_actions):
            for s_n in range(n_states):
                for a_n in range(n_actions):
                    P_pi_star[s * n_actions + a, s_n * n_actions + a_n] = \
                        P_0[s, s_n, a] if a_n == pi_star[s_n] else 0
    return P_pi_star
# enddef


def calculate_I_R(env, pi_d, delta_s_array_orig, H_set):
    n_states = env.n_states
    n_actions = env.n_actions
    R = env.reward

    delta_s_eps_h_s_array_diff = []

    delta_s_array = np.array(delta_s_array_orig)

    I_pi_star = calculate_I_pi_star(env, pi_d)
    P_pi_star = calculate_P_pi_star(env, pi_d)
    I = np.eye(n_states * n_actions)

    for i, H in enumerate(H_set):
        accumulator = get_A_local_h(env, P_pi_star, I, H=H)

        A_local = (I_pi_star - I) @ (accumulator)

        A_x_local = A_local @ (R.flatten())

        for s in range(n_states):
            s_a_array = []
            for a in range(n_actions):
                if pi_d[s] != a:
                    s_a_array.append(delta_s_array[s] - A_x_local[s * n_actions + a])
            delta_s_eps_h_s_array_diff.append(max(np.hstack(s_a_array)))

    I_R = - (1 / len(H_set)) * (1 / n_states) * np.sum(np.maximum(0, delta_s_eps_h_s_array_diff))
    return I_R
#enddef

def get_env_with_new_R(env_orig, reward):
    env_new = copy.deepcopy(env_orig)
    env_new.reward = copy.deepcopy(reward)
    return env_new
#ednndef

def input_dict_from_file_and_plot_steps_and_exp_reward(dict_file_q, n_file=10,
                                                       out_folder_name="out_folder", each_number=50, t=100):
    if not os.path.isdir(out_folder_name):
        os.makedirs(out_folder_name)

    # for key, value in dict_file_q.items():
    #     dict_file_q[key] = value / (n_file)


    plt.figure(3, figsize=fig_size)

    up_to_power_2 = 15

    array_index_to_plot = [0]
    for i in range(0, up_to_power_2+1):
        array_index_to_plot.append(int(2**i))

    keys = [
    "R_landmark_state_design_env=room_n_states=50_len_active_set_chosen=6_expected_reward",
            "R_reward_design_env=room_n_states=50_len_active_set_chosen=4_expected_reward",
            "R_reward_design_env=room_n_states=50_len_active_set_chosen=6_expected_reward",
            "Reward_all_state_design_expected_reward",
            "R_original_expected_reward",
            "R_potential_shaping_expected_reward",
            "R_landmark_handpicked_env=room_n_states=50_len_active_set_chosen=6_expected_reward",
            "Pot_craft_expected_reward"
            ]

    for key in keys:
        print("=========")
        print(key)

        if key == "R_landmark_state_design_env=room_n_states=50_len_active_set_chosen=6_expected_reward":
            plt.errorbar(range(0, up_to_power_2+2),
                     dict_file_q[key][array_index_to_plot],
                     label=r"$\textsc{ExpRD} \ (B=5, \lambda\to\infty)$",
                     color="#808080", marker="x", ls=":", markersize=10, lw=2.5,
                     yerr=dict_file_q["SE_craft_design"][array_index_to_plot])

        elif key == "R_reward_design_env=room_n_states=50_len_active_set_chosen=4_expected_reward":
            # picked_states = dict_file[]
            plt.errorbar(range(0, up_to_power_2+2),
                     dict_file_q[key][array_index_to_plot],
                     label=r"$\textsc{ExpRD} \ (B=3, \lambda=0 )$",
                     color='#43BFC7', marker="d", ls='-.', lw=2.5,markersize=10,
                     yerr=dict_file_q["SE_Reward_design_3"][array_index_to_plot])


        elif key == "R_reward_design_env=room_n_states=50_len_active_set_chosen=6_expected_reward":
            # picked_states = dict_file[]
            plt.errorbar(range(0, up_to_power_2+2),
                     dict_file_q[key][array_index_to_plot],
                     label=r"$\textsc{ExpRD} \ (B=5, \lambda=0)$",
                     color='#69BE28', marker=">", lw=4,markersize=10,
                     yerr=dict_file_q["SE_Reward_design"][array_index_to_plot])

        elif key == "Reward_all_state_design_expected_reward":
            plt.errorbar(range(0, up_to_power_2+2),
                     dict_file_q[key][array_index_to_plot],
                     label=r"$\textsc{ExpRD}\ (B=|S|, \lambda=0)$",
                     color='#FF8849', marker="<", ls=":", lw=4, markersize=10,
                     yerr=dict_file_q["SE_Reward_design_all"][array_index_to_plot])


        elif key =="R_original_expected_reward":
            plt.errorbar(range(0, up_to_power_2+2),
                     dict_file_q[key][array_index_to_plot],
                     label=r"$\textsc{Orig}$",
                     color=color_codes[0], marker="s", ls="-.", lw=4, markersize=10,
                     yerr=dict_file_q["SE_Reward_orig"][array_index_to_plot])
        elif key == "R_potential_shaping_expected_reward":
            plt.errorbar(range(0, up_to_power_2+2),
                     dict_file_q[key][array_index_to_plot],
                     label=r"$\textsc{PBRS}$",
                     color='#0a81ab', marker=".", ls="-.", markersize=18, lw=4,
                     yerr=dict_file_q["SE_Reward_pot"][array_index_to_plot])

        elif key == "Pot_craft_expected_reward":
            plt.errorbar(range(0, up_to_power_2+2),
                     dict_file_q[key][array_index_to_plot],
                     label=r"$\textsc{PBRS-SEC}$",
                     color='#f0e130', marker="x", ls=":", markersize=18, lw=4,
                     yerr=dict_file_q["SE_pot_craft"][array_index_to_plot])



        elif key == "R_landmark_handpicked_env=room_n_states=50_len_active_set_chosen=6_expected_reward":
            plt.errorbar(range(0, up_to_power_2+2),
                     dict_file_q[key][array_index_to_plot],
                     label=r"$\textsc{Craft}\ (B=5)$ ",
                     color="#8b0000", marker="^", ls="-.", markersize=10, lw=2.5,
                     yerr=dict_file_q["SE_craft"][array_index_to_plot])


    # print(dict_file)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.57), ncol=2, fancybox=False, shadow=False)
    plt.ylabel(r"Expected reward")
    plt.xlabel(r'Episode')
    plt.xticks(range(0, up_to_power_2+2)[::2], ["$0$", "$2^0$", "$2^1$", "$2^2$", "$2^3$", "$2^4$",
                              "$2^5$", "$2^6$", "$2^7$", "$2^8$", "$2^9$", "$2^{10}$"
                              , "$2^{11}$", "$2^{12}$", "$2^{13}$", "$2^{14}$", "$2^{15}$"][::2])
    # plt.ylim(ymin=0.0)
    # plt.xticks(range(len(dict_averaged_acumulator["reward_design_facility_steps_to_opt_det_policy_arr_lambda_1"])),
    #            range(1,
    #                  len(dict_averaged_acumulator["reward_design_facility_steps_to_opt_det_policy_arr_lambda_1"]) + 1,
    #                  1))

    outFname = os.getcwd() + "/" + out_folder_name + "/fourroom_convergence_fixed_B.pdf"
    plt.savefig(outFname, bbox_inches='tight')
    pass
#enddef

def input_from_file(input_dir_path_q_learning_convergence_numbers, n_file=10, t=100):
    if not os.path.isdir(out_folder_name):
        os.makedirs(out_folder_name)

    dict_file_q ={}
    std_matrix_orig = []
    std_matrix_pot = []
    std_matrix_pot_craft = []
    std_matrix_dsgn = []
    std_matrix_dsgn_all = []
    std_matrix_craft = []
    std_matrix_craft_design = []
    std_matrix_design_3 = []

    file_path = input_dir_path_q_learning_convergence_numbers
    expname = '/convergence_'
    for file in range(1, n_file + 1):
        with open(file_path + expname + str(file) + '.txt') as f:
            # with open(file_name) as f:
            print(file_path + expname + str(file) + '.txt')
            for line in f:
                read_line = line.split()
                if read_line[0] != '#':
                    if read_line[0] == 'initStates' or read_line[0] == 'active_state_opt_states' or read_line[
                        0] == 'sgd_state_opt_states' or read_line == 'template_list_for_initStates':
                        dict_file_q[read_line[0]] = np.array(list(map(int, read_line[1:t])))
                    elif read_line[0] in dict_file_q.keys():
                        dict_file_q[read_line[0]] += np.array(list(map(float, read_line[1:t])))
                    else:
                        dict_file_q[read_line[0]] = np.array(list(map(float, read_line[1:t])))
                    if read_line[0] == "R_original_expected_reward":
                        std_matrix_orig.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "R_potential_shaping_expected_reward":
                        std_matrix_pot.append(np.array(list(map(float, read_line[1:t]))))

                    if read_line[0] == "Pot_craft_expected_reward":
                        std_matrix_pot_craft.append(np.array(list(map(float, read_line[1:t]))))

                    if read_line[0] == "R_landmark_handpicked_env=room_n_states=50_len_active_set_chosen=6_expected_reward":
                        std_matrix_dsgn.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "Reward_all_state_design_expected_reward":
                        std_matrix_dsgn_all.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "R_landmark_handpicked_env=room_n_states=50_len_active_set_chosen=6_expected_reward":
                        std_matrix_craft.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "R_landmark_state_design_env=room_n_states=50_len_active_set_chosen=6_expected_reward":
                        std_matrix_craft_design.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "R_reward_design_env=room_n_states=50_len_active_set_chosen=4_expected_reward":
                        std_matrix_design_3.append(np.array(list(map(float, read_line[1:t]))))


    std_orig = np.std(std_matrix_orig, axis=0)
    std_pot = np.std(std_matrix_pot, axis=0)
    std_pot_craft = np.std(std_matrix_pot_craft, axis=0)
    std_dsgn = np.std(std_matrix_dsgn, axis=0)
    std_dsgn_all = np.std(std_matrix_dsgn_all, axis=0)
    std_craft = np.std(std_matrix_craft, axis=0)
    std_craft_design = np.std(std_matrix_craft_design, axis=0)
    std_design_3 = np.std(std_matrix_design_3, axis=0)

    SE_orig = std_orig / np.sqrt(len(std_matrix_orig))
    SE_pot = std_pot / np.sqrt(len(std_matrix_pot))
    SE_pot_craft = std_pot_craft / np.sqrt(len(std_matrix_pot_craft))
    SE_dsgn = std_dsgn / np.sqrt(len(std_matrix_dsgn))
    SE_dsgn_all = std_dsgn_all / np.sqrt(len(std_matrix_dsgn_all))
    SE_craft = std_craft / np.sqrt(len(std_matrix_craft))
    SE_craft_design = std_craft_design / np.sqrt(len(std_matrix_craft_design))
    SE_design_3 = std_design_3 / np.sqrt(len(std_matrix_design_3))



    for key, value in dict_file_q.items():
        dict_file_q[key] = value / (n_file)

    dict_file_q["SE_Reward_orig"] = np.array(SE_orig)
    dict_file_q["SE_Reward_pot"] = np.array(SE_pot)
    dict_file_q["SE_Reward_design"] = np.array(SE_dsgn)
    dict_file_q["SE_Reward_design_all"] = np.array(SE_dsgn_all)
    dict_file_q["SE_craft"] = np.array(SE_craft)
    dict_file_q["SE_craft_design"] = np.array(SE_craft_design)
    dict_file_q["SE_Reward_design_3"] = np.array(SE_design_3)
    dict_file_q["SE_pot_craft"] = np.array(SE_pot_craft)
    return dict_file_q
#enndef


if __name__ == "__main__":

    upper_c_for_delt_s = None
    n_averaged = int(sys.argv[1])

    input_dir_path_q_learning_convergence_numbers = \
        "results/Q_convergence_room_False_const_middle__sub_optimal_IR/"
    out_folder_name = "output"

    each_number = 800
    t = int(90000)

    tol = 1e-10

    dict_file = input_from_file(
        input_dir_path_q_learning_convergence_numbers, n_file=n_averaged, t=t)
    # dict_alg_5,SE_5 = input_from_file(
    #     input_dir_path_q_learning_convergence_numbers_5, n_file=n_averaged_5, t=t)

    for key in dict_file:
        if "SE" in key:
            print(dict_file[key])
    # exit(9)

    env_args = {
        "gridsizefull": 7,
        "R_max": 10,
        "gamma": 0.95,
        "terminalState": 1,
        "randomMoveProb": 0.1,
    }

    env_orig = room.Environment(env_args)

    Q_orig, V_ori, pi_d_ori, pi_s_ori = MDPSolver.valueIteration(env_orig, env_orig.reward)

    exp_reward = np.dot(env_orig.InitD, V_ori)

    exp_reward_20 = exp_reward*0.2
    exp_reward_50 = exp_reward*0.5
    exp_reward_80 = exp_reward*0.8



    dict_to_plot = {}

    dict_to_plot.update(dict_file)
    # dict_to_plot.update(dict_alg_2)
    # dict_to_plot.update(dict_alg_3)
    # dict_to_plot.update(dict_alg_4)
    # # dict_to_plot.update(dict_alg_5)
    # dict_to_plot["SE_Reward_orig"] = SE_1
    # dict_to_plot["SE_Reward_pot"] = SE_2
    # dict_to_plot["SE_Reward_design"] = SE_3
    # dict_to_plot["SE_Reward_design_all"] = SE_4
    # # dict_to_plot["SE_Reward_pot_cont"] = SE_5
    # print(SE_5)
    # exit(0)
    # print(dict_to_plot["R_potential_shaping_expected_reward"])
    # exit(0)



    # exit(0)


    input_dict_from_file_and_plot_steps_and_exp_reward(dict_to_plot,
                                                       n_file=n_averaged, out_folder_name=out_folder_name,
                                                       each_number=each_number, t=t)

    # exit(0)

    #
    # dict_convergence_numbers_q_learning = input_from_file_q_convergence_and_plot(dir_path_q_learning_convergence_numbers, n_file=n_averaged,
    #                                                       outfile_name="results/plots_chain_{}".format(is_delta_s_const))
    exit(0)

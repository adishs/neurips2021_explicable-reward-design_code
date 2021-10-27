import numpy as np
import copy
import sys
import itertools as it
import os
sys.path.append('../')
sys.path.append('../code-attacker/src')
import MDPSolver
import fourroom_env as room
import q_learning

import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib as mpl
mpl.rcParams['font.serif'] = ['times new roman']
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']

# color_codes = ["#C64C23", "#0A02FB", "#223206", "#7E3391", "#E802FB", "#040004", "#7E3391", "#E802FB", "#040004"]
color_codes = [ 'r', 'g', 'b', '#F08080', "#8B0000", '#E802FB']
color_for_orig = "#8b0000"
color_for_pot = "#F08080"

mpl.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 32})
mpl.rc('legend', **{'fontsize': 22})
mpl.rc('text', usetex=True)
fig_size = [6.4, 4.8]

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

def append_to_accumulator(main_accumulator, acc_tmp):
    for key in acc_tmp:
        main_accumulator[key] = acc_tmp[key]
    return main_accumulator
    #enddef



def get_convergence_numbers_q_learning(dict_file, env_orig, pi_target_d, pi_target_s,
                                       n_epoch=1000, horizon=500, epsilon=0.1, alpha=0.5,
                                              outfile_name="out_file", n_averaged=5, tol=1e-10):
    # if not os.path.isdir(outfile_name):
    #     os.makedirs(outfile_name)

    max_episode = n_epoch
    max_step = horizon

    # sort to maintain increase order of the choosen set size
    all_keys = list(dict_file.keys())
    sorted_keys_by_length = sorted(all_keys, key=len)

    for i in range(1, n_averaged+1):
        dict_accumulator_main = {}
        for key in sorted_keys_by_length:
            if "R_original" in key or \
                    "R_potential_shaping" in key or \
                    "R_reward_design_env=room_n_states=50_len_active_set_chosen=6" in key or \
                     "R_reward_design_env=room_n_states=50_len_active_set_chosen=4" in key \
                or "Reward_all_state_design" in key: # execute only for the |Z|=5
                print("\n================")
                print(key)
                print("================\n")
                R_input = dict_file[key].reshape(env_orig.n_states, env_orig.n_actions)
                env_R_input = get_env_with_new_R(env_orig, R_input)
                dict_tmp = q_learning.q_learning(
                    env=env_R_input, max_episode=max_episode, max_step=max_step, epsilon=epsilon, alpha=alpha,
                    env_orig=env_orig,
                    pi_t=pi_target_d, name=key, tol=tol
                )
                dict_accumulator_main = append_to_accumulator(dict_accumulator_main, dict_tmp)

        write_into_file(dict_accumulator_main, exp_iter=str(i), out_folder_name=outfile_name,
                        out_file_name="convergence")
    return
#enddef


if __name__ == "__main__":

    input_dir_name = sys.argv[1]
    n_averaged = int(sys.argv[2])

    is_delta_s_const = False
    n_epoch = int(2**15+1)
    horizon = 50
    epsilon = 0.1
    alpha = 0.5

    filename = "information_is_delta_s_const_{}_1.txt".format(is_delta_s_const)
    dir_path ="results/" + input_dir_name

    tol = 1e-10

    dict_file = input_from_file(dir_path, filename)
    # print(dict_file["H_set"])

    env_orig = room.Environment(dict_file)

    Q_orig, V_orig, pi_d_orig, pi_s_orig = MDPSolver.valueIteration(env_orig, env_orig.reward, tol=tol)
    pi_target_d = copy.deepcopy(pi_d_orig)
    pi_target_s = copy.deepcopy(pi_s_orig)


    get_convergence_numbers_q_learning(dict_file, env_orig, pi_target_d, pi_target_s,
                                       n_epoch=n_epoch, horizon=horizon, epsilon=epsilon, alpha=alpha,
                                       outfile_name="Q_learning_data/Q_learning_{}".format(input_dir_name),
                                       n_averaged=n_averaged, tol=1e-10)

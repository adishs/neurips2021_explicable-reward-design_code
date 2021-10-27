import numpy as np
import copy
import sys
import itertools as it
import os
sys.path.append('../')
sys.path.append('../code-attacker/src')
import MDPSolver
import fourroom_env as room


from fourroom_reward_design_exprd import *


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

def get_H_set(env):
    one_over_one_minus_gamma_int = int(np.round(1/(1-env.gamma), 5))
    H_set = range(one_over_one_minus_gamma_int//2, one_over_one_minus_gamma_int+1)
    # h = copy.deepcopy(n_states)
    #
    # while h > 4:
    #
    #     H_set.append(h)
    #     h//=2

    return H_set
#enddef

def get_reward_landmark(env, pi_s):
    reward = np.zeros((env.n_states, env.n_actions))

    for s in range(env.n_states-env.terminal_state):
        for a in range(env.n_actions):
            if s in env.gate_states:
                if pi_s[s, a] == 0:
                    reward[s, a] = -1
                else:
                    reward[s, a] = 1
    if env.terminal_state == 0:
        reward[-1, :] = 10.0
    else:
        reward[-2, :] = 10.0

    return reward
# enndef

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

def get_landmark_reward_for_room(env_orig, pi_d, n_landmarks, R_scalar_for_landmark,
                                  env_name="", exp_name="", exp_iter=1, out_file_name=""):
    n_states = env_orig.n_states
    reward_landmark = copy.deepcopy(env_orig.reward)
    # write_R_into_file(env_orig, reward_landmark, s_active_set_chosen=states_for_landmark, exp_iter=exp_iter,
    #                   env_name=env_name, exp_name=exp_name, out_file_name=out_file_name)

    states_for_landmark_goal = env_orig.goal_state
    ########## OUTPUT into FILE###################
    accumulator_for_file["R_{}_env={}_n_states={}_len_active_set_chosen={}".format(exp_name, env_name, n_states,
                                                                   len(states_for_landmark_goal))] = reward_landmark.flatten()
    if n_landmarks == 1:
        return reward_landmark, states_for_landmark_goal
    else:
        incremental_landmark_states = [env_orig.goal_state[0]]
        # states_for_landmark = np.array([0, 10, 5, 15]) # n_states = 20
        states_for_landmark = np.concatenate((env_orig.gate_states, [32]))
        for i, s in enumerate(states_for_landmark):
            if i == n_nonzero_states-1:
                break
            else:
                incremental_landmark_states.append(s)
                for a in range(env_orig.n_actions):
                    if a != pi_d[s]:
                        reward_landmark[s, a] = - R_scalar_for_landmark
                    else:
                        reward_landmark[s, a] = R_scalar_for_landmark

            accumulator_for_file[
                "R_{}_env={}_n_states={}_len_active_set_chosen={}".format(exp_name, env_name, n_states,
                                                                          len(
                                                                              incremental_landmark_states))] = reward_landmark.flatten()
    # states_for_landmark = [0]
    # len_of_cut = n_states // (n_landmarks)
    # middle_state = n_states//2
    # counter_len_cut = 0
    #
    # for i in range(n_landmarks-2):
    #     if i % 2 == 0:
    #         states_for_landmark.append(middle_state + counter_len_cut * len_of_cut)
    #     else:
    #         counter_len_cut += 1
    #         states_for_landmark.append(middle_state - counter_len_cut * len_of_cut)
    #
    # # sorted_states_for_landmark = sorted(states_for_landmark)
    # incremental_landmark_states = np.array([])
    # for s in states_for_landmark:
    #     incremental_landmark_states = np.concatenate((incremental_landmark_states, [s]))
    #     for a in range(env_orig.n_actions):
    #         if a != pi_d[s]:
    #             reward_landmark[s, a] = - R_scalar_for_landmark
    #         else:
    #             reward_landmark[s, a] = R_scalar_for_landmark
    #
    #     ########## OUTPUT into FILE###################
    #     accumulator_for_file["R_{}_env={}_n_states={}_s_active_set_chosen={}".format(exp_name, env_name, n_states,
    #                                                                                  "_".join(str(i) for i in (
    #                                                                                      incremental_landmark_states)))] = reward_landmark.flatten()
    #     # write_R_into_file(env_orig, reward_landmark, s_active_set_chosen=incremental_landmark_states, exp_iter=exp_iter,
    #     #                   env_name=env_name, exp_name=exp_name, out_file_name=out_file_name)
    # accumulator_for_file[
    # "R_{}_env={}_n_states={}_s_active_set_chosen={}".format(exp_name, env_name, n_states,
    #                                                         "_".join(str(int(i)) for i in (
    #                                                             incremental_landmark_states)))] = reward_landmark.flatten()
    accumulator_for_file["s_active_set_chosen_by_hand"] = incremental_landmark_states
    return reward_landmark, incremental_landmark_states
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

def get_important_states_by_shaping(env_orig, pi_d, pi_s, R_max, H_set, s_active=None,
                                      delta_s_array=None, C_1=1.0, C_2=1.0, n_nonzero_states=1, candidate_states=None,
                                      dict_s_opt_actions_arr=None, state_only_reward_flag=False, is_delta_s_const=False,
                                    upper_c_for_delt_s=None, tol=1e-10,
                                        env_name="", exp_name="", exp_iter=1, out_file_name="rewards"):
    n_states = env_orig.n_states

    if s_active is None:
        s_active = env_orig.goal_state  # last or second last state as goal

    s_active_set_chosen = copy.deepcopy(list(env_orig.goal_state))


    obj_value, R_goal = reward_design_model_based(env_orig, pi_d=pi_d, pi_s=pi_s, R_max=R_max, H_set=H_set,
                                                    s_active=s_active_set_chosen,
                                                    delta_s_array=delta_s_array, C_1=C_1, C_2=C_2,
                                                    dict_s_opt_actions_arr=dict_s_opt_actions_arr,
                                                    state_only_reward_flag=state_only_reward_flag,  is_delta_s_const=is_delta_s_const, upper_c_for_delt_s=upper_c_for_delt_s, tol=tol)
    ########## OUTPUT into FILE###################
    accumulator_for_file["R_{}_env={}_n_states={}_len_active_set_chosen={}".format(exp_name, env_name, n_states,
                                                                   len(s_active_set_chosen))] = R_goal.flatten()

    for i in range(1, n_nonzero_states):
        min_value = np.inf
        state_chosen = None
        R_shaped = None
        for s in candidate_states:
            if s not in s_active_set_chosen:
                s_active_candidate = np.concatenate((s_active_set_chosen, [s]))
                print("===================")
                print("S_active_chosen: ", s_active_set_chosen)
                print("S_candidate: ", s)
                obj_value, R_sol = reward_design_model_based(env_orig, pi_d=pi_d, pi_s=pi_s, R_max=R_max, H_set=H_set,
                                                          s_active=s_active_candidate,
                                                          delta_s_array=delta_s_array, C_1=C_1, C_2=C_2,
                                                          dict_s_opt_actions_arr=dict_s_opt_actions_arr,
                                                         state_only_reward_flag=state_only_reward_flag,  is_delta_s_const=is_delta_s_const, upper_c_for_delt_s=upper_c_for_delt_s, tol=tol)
                print("obj_value: ", obj_value)
                print("===================")

                if min_value > obj_value:
                    min_value = copy.deepcopy(obj_value)
                    state_chosen = copy.deepcopy(s)
                    R_shaped = copy.deepcopy(R_sol)
        s_active_set_chosen.append(state_chosen)
        ########## OUTPUT into FILE###################
        accumulator_for_file["R_{}_env={}_n_states={}_len_active_set_chosen={}".format(exp_name, env_name, n_states,
                                                                                       len(
                                                                                           s_active_set_chosen))] = R_shaped.flatten()
    accumulator_for_file["s_active_set_chosen_by_shaping"] = s_active_set_chosen
    return s_active_set_chosen, R_shaped
# enddef


def shaping_over_landmark_states(env_orig, pi_d, pi_s, R_max, H_set,landmark_states=None,
                                      delta_s_array=None, C_1=1.0, C_2=1.0,
                                      dict_s_opt_actions_arr=None, state_only_reward_flag=False, is_delta_s_const=False,
                                 upper_c_for_delt_s=None, tol=1e-10,
                                        env_name="", exp_name="", exp_iter=1, out_file_name="rewards"):
    n_states = env_orig.n_states

    incremental_randmark_state_array = []
    for s in landmark_states:
        incremental_randmark_state_array = np.concatenate((incremental_randmark_state_array, [s]))
        print("===================")
        print("S_active_chosen: ", incremental_randmark_state_array)
        print("S_candidate: ", s)
        obj_value, R_sol = reward_design_model_based(env_orig, pi_d=pi_d, pi_s=pi_s, R_max=R_max, H_set=H_set,
                                                     s_active=incremental_randmark_state_array,
                                                     delta_s_array=delta_s_array, C_1=C_1, C_2=C_2,
                                                     dict_s_opt_actions_arr=dict_s_opt_actions_arr,
                                                     state_only_reward_flag=state_only_reward_flag,  is_delta_s_const=is_delta_s_const, upper_c_for_delt_s=upper_c_for_delt_s, tol=tol)
        print("obj_value: ", obj_value)
        print("===================")
        ########## OUTPUT into FILE###################
        accumulator_for_file["R_{}_env={}_n_states={}_len_active_set_chosen={}".format(exp_name, env_name, n_states,
                                                                                       len(
                                                                                           incremental_randmark_state_array))] = R_sol.flatten()
        accumulator_for_file["s_active_set_chosen_by_landmark_shaping"] = np.array(incremental_randmark_state_array, dtype=int)
    return incremental_randmark_state_array, R_sol
    # enddef

def get_env_with_new_R(env_orig, reward):
    env_new = copy.deepcopy(env_orig)
    env_new.reward = copy.deepcopy(reward)
    return env_new
#ednndef

def facility_function(env, S, Z):
    sum_over_s = 0
    for s in S:
        sum_over_s += min([env.distance_fun(s, z) for z in Z])
    return sum_over_s
#enddef

def get_important_states_by_shaping_and_facility(env_orig, pi_d, pi_s, R_max, H_set, s_active=None,
                                      delta_s_array=None, C_1=1.0, C_2=1.0, lmbd=1,  n_nonzero_states=1, candidate_states=None,
                                      dict_s_opt_actions_arr=None, state_only_reward_flag=False, is_delta_s_const=False,
                                                 upper_c_for_delt_s=None, tol=1e-10,
                                        env_name="", exp_name="", exp_iter=1, out_file_name="rewards"):
    n_states = env_orig.n_states
    S_set = range(env_orig.n_states-env_orig.terminal_state)

    if s_active is None:
        s_active = env_orig.goal_state  # goal

    s_active_set_chosen = copy.deepcopy(list(env_orig.goal_state))

    obj_value_IR, R_goal = reward_design_model_based(env_orig, pi_d=pi_d, pi_s=pi_s, R_max=R_max, H_set=H_set,
                                                    s_active=s_active_set_chosen,
                                                    delta_s_array=delta_s_array, C_1=C_1, C_2=C_2,
                                                    dict_s_opt_actions_arr=dict_s_opt_actions_arr,
                                                    state_only_reward_flag=state_only_reward_flag,  is_delta_s_const=is_delta_s_const, upper_c_for_delt_s=upper_c_for_delt_s, tol=tol)
    ########## OUTPUT into FILE###################
    accumulator_for_file["R_{}_env={}_n_states={}_lmbd={}_len_active_set_chosen={}".format(exp_name, env_name, n_states, lmbd,
                                                                                   len(
                                                                                       s_active_set_chosen))] = R_goal.flatten()

    for i in range(1, n_nonzero_states):
        min_value = np.inf
        state_chosen = None
        R_shaped = None
        for s in candidate_states:
            if s not in s_active_set_chosen:
                s_active_candidate = np.concatenate((s_active_set_chosen, [s]))
                print("===================")
                print("S_active_chosen: ", s_active_set_chosen)
                print("S_candidate: ", s)
                obj_value_IR, R_sol = reward_design_model_based(env_orig, pi_d=pi_d, pi_s=pi_s, R_max=R_max, H_set=H_set,
                                                          s_active=s_active_candidate,
                                                          delta_s_array=delta_s_array, C_1=C_1, C_2=C_2,
                                                          dict_s_opt_actions_arr=dict_s_opt_actions_arr,
                                                         state_only_reward_flag=state_only_reward_flag,  is_delta_s_const=is_delta_s_const, upper_c_for_delt_s=upper_c_for_delt_s, tol=tol)
                print("obj_value: ", obj_value_IR)
                print("===================")

                obj_value = obj_value_IR + lmbd * facility_function(env_orig, S_set, s_active_candidate)

                if min_value > obj_value:
                    min_value = copy.deepcopy(obj_value)
                    state_chosen = copy.deepcopy(s)
                    R_shaped = copy.deepcopy(R_sol)
        s_active_set_chosen.append(state_chosen)
        ########## OUTPUT into FILE###################
        # write_R_into_file(env_0, R_shaped, s_active_set_chosen=s_active_set_chosen, exp_iter=exp_iter,
        accumulator_for_file[
            "R_{}_env={}_n_states={}_lmbd={}_len_active_set_chosen={}".format(exp_name, env_name, n_states, lmbd,
                                                                              len(
                                                                                  s_active_set_chosen))] = R_shaped.flatten()

    accumulator_for_file["s_active_set_chosen_by_shaping_and_facility_lmbd={}".format(np.round(lmbd, 5))] = s_active_set_chosen
    return s_active_set_chosen, R_shaped
# enddef

def get_potential_based_reward_craft(env, R_craft, tol=1e-10):
    R_pot = copy.deepcopy(env.reward)
    Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, R_craft, tol=tol)
    V = np.round(V, 10)
    for s in range(env.n_states):
        for a in range(env.n_actions):
            sum_over_next_states = 0
            for n_s in range(env.n_states):
                sum_over_next_states += env.gamma * env.T[s, n_s, a] * V[n_s]
            R_pot[s, a] += sum_over_next_states - V[s]

    return R_pot
#enddef



if __name__ == "__main__":

    #Parameters

    dict_s_opt_actions_arr = {}
    n_nonzero_states = 6
    C_1 = 1
    C_2 = 0 #
    R_max = 10
    state_only_reward_flag = False
    is_delta_s_const = False
    upper_c_for_delt_s = None
    tol = 1e-10
    lmbd_array = [1e-3, 1, int(1e10)]


    ##############

    # print("a=", np.genfromtxt('results/designed_rewards/R_designed_env=chain_n_states=16_s_active_set_chosen=14_0_3_5_6_iter=1.txt'))
    # exit(0)

    env_args = {
        "gridsizefull": 7,
        "R_max": R_max,
        "gamma": 0.95,
        "terminalState": 1,
        "randomMoveProb": 0.1,
    }

    env_orig = room.Environment(env_args)
    for key in env_args:
        accumulator_for_file[key] = [env_args[key]]

    accumulator_for_file["R_original"] = env_orig.reward.flatten()

    accumulator_for_file["R_potential_shaping"] = get_potential_based_reward(env_orig, tol).flatten()
    H_set = [1, 4, 8, 16, 32] #get_H_set(env_orig) # (1/1-gamma)//2, ...., 1/(1-gamma)
    accumulator_for_file["H_set"] = H_set

    Q_orig, V_orig, pi_d_orig, pi_s_orig = MDPSolver.valueIteration(env_orig, env_orig.reward, tol=tol)

    # print(get_landmark_reward_for_chain(env_orig, pi_d_orig, 1, R_scalar_for_landmark=2))
    # exit(0)

    #==============================
    pi_target_d = copy.deepcopy(pi_d_orig)
    pi_target_s = copy.deepcopy(pi_s_orig)
    #==============================

    #===========================================
    delta_s_array_orig = get_delta_s_given_policy(env_orig, pi_target_d, pi_target_s, tol=tol)
    if is_delta_s_const:
        upper_c_for_delt_s = sorted(delta_s_array_orig)[len(delta_s_array_orig)//2]
    # ===========================================

    handpicked_states = np.concatenate((env_orig.goal_state, env_orig.gate_states))

    _, R_design_handpicked_states = reward_design_model_based(env_orig, pi_target_d, pi_target_s, R_max, H_set,s_active=handpicked_states,
                                      delta_s_array=delta_s_array_orig, C_1=C_1, C_2=C_2,
                                      dict_s_opt_actions_arr=dict_s_opt_actions_arr,
                                        state_only_reward_flag=False,  is_delta_s_const=is_delta_s_const, upper_c_for_delt_s=upper_c_for_delt_s, tol=1e-10)
    print(R_design_handpicked_states)
    # exit(0)


    s_active = None




    candidate_states = range(0, env_orig.n_states-env_orig.terminal_state - 1)
    states_active, R_design = get_important_states_by_shaping(env_orig, pi_d=pi_target_d, pi_s=pi_target_s, R_max=R_max, H_set=H_set, s_active=s_active,
                                                delta_s_array=delta_s_array_orig, C_1=C_1, C_2=C_2, n_nonzero_states=n_nonzero_states,
                                                candidate_states=candidate_states,
                                                dict_s_opt_actions_arr=dict_s_opt_actions_arr,
                                                state_only_reward_flag=state_only_reward_flag, is_delta_s_const=is_delta_s_const, upper_c_for_delt_s=upper_c_for_delt_s, tol=tol,
                                                env_name="room", exp_name="reward_design", exp_iter=1, out_file_name="designed_rewards")

    # =============== DESIGNED ====================================

    # M_design = (M_orig[0], M_orig[1], R_design, M_orig[3], M_orig[4], M_orig[5])
    env_design = get_env_with_new_R(env_orig, R_design)

    Q_design, V_design, pi_d_design, pi_s_design = MDPSolver.valueIteration(env_design, env_design.reward, tol=tol)

    delta_s_array_design = get_delta_s_given_policy(env_design, pi_target_d, pi_target_s, tol=tol)

    IR_design = calculate_I_R(env_design, pi_d_design, delta_s_array_orig, H_set)


    print("==================DESIGNED====================================")
    print("Designed delta_inf = ", min(delta_s_array_design ))
    print("states picked: ", states_active)
    print("IR design = ", IR_design)
    print(R_design)
    print("======================================================")

    ########### DESIGN + facility ##################


    # for lmbd in lmbd_array:
    #
    #     states_active_shaping_and_facility, R_design_shaping_and_facility = get_important_states_by_shaping_and_facility(env_orig,
    #                                                 pi_d=pi_target_d, pi_s=pi_target_s, R_max=R_max, H_set=H_set, s_active=s_active,
    #                                                 delta_s_array=delta_s_array_orig, C_1=C_1, C_2=C_2, lmbd=lmbd, n_nonzero_states=n_nonzero_states,
    #                                                 candidate_states=candidate_states,
    #                                                 dict_s_opt_actions_arr=dict_s_opt_actions_arr,
    #                                                 state_only_reward_flag=state_only_reward_flag,  is_delta_s_const=is_delta_s_const, upper_c_for_delt_s=upper_c_for_delt_s, tol=tol,
    #                                                 env_name="room", exp_name="reward_design_facility", exp_iter=1, out_file_name="designed_rewards")

    # print(R_sol)
    # print("states_active", states_active)
    # print("R_design", R_design)
    # print("=====================")
    # print("states_active_shaping_and_facility", states_active_shaping_and_facility)
    # print("R_design_shaping_and_facility", R_design_shaping_and_facility)
    # print("=====================")
    # exit(0)




    #=============== POTENTIAL ====================================

    R_pot = get_potential_based_reward(env_orig, tol)
    # print(R_pot)

    # M_pot = (M_orig[0], M_orig[1], R_pot, M_orig[3], M_orig[4], M_orig[5])
    env_pot = get_env_with_new_R(env_orig, R_pot)

    Q_pot, V_pot, pi_d_pot, pi_s_pot = MDPSolver.valueIteration(env_pot, env_pot.reward, tol=tol)

    delta_s_array_pot = get_delta_s_given_policy(env_pot, pi_target_d, pi_target_s, tol=tol)

    IR_pot = calculate_I_R(env_pot, pi_d_pot, delta_s_array_orig, H_set)
    # print(IR_pot)
    # exit(0)

    print("=====================POTENTIAL=================================")
    print("Potential delta_inf = ", min(delta_s_array_pot ))
    print("IR_pot  = ", IR_pot)
    print("======================================================")



    #=============== LANDMARK ====================================

    R_landmark, landmark_states = get_landmark_reward_for_room(env_orig, pi_d_orig,
                                                                n_landmarks=n_nonzero_states, R_scalar_for_landmark=1,
                                                                env_name="room", exp_name="landmark_handpicked", exp_iter=1,
                                                                out_file_name="landmark_rewards")

    # M_landmark = (M_orig[0], M_orig[1], R_landmark, M_orig[3], M_orig[4], M_orig[5])
    # env_landmark = env_chain.Environment("", M_landmark)
    env_landmark = get_env_with_new_R(env_orig, R_landmark)

    Q_landmark, V_landmark, pi_d_landmark, pi_s_landmark = MDPSolver.valueIteration(env_landmark,
                                                                                    env_landmark.reward, tol=tol)

    delta_s_array_landmark = get_delta_s_given_policy(env_landmark, pi_target_d, pi_target_s, tol=tol)

    IR_landmark = calculate_I_R(env_landmark, pi_d_landmark, delta_s_array_orig, H_set)

    print("=====================LANDMARK=================================")
    print("Landmark delta_inf = ", min(delta_s_array_landmark ))
    print("IR_landmark  = ", IR_landmark)
    print("landmark_states = ", landmark_states)
    print("======================================================")

    R_pot_craft = get_potential_based_reward_craft(env_orig, R_landmark)

    accumulator_for_file["Pot_craft"] = R_pot_craft.flatten()

    # ===============LANDMARK STATE DESIGN========================
    s_active_landmark = copy.deepcopy(landmark_states)
    _, R_landmark_state_design = shaping_over_landmark_states(env_orig, pi_target_d, pi_target_s, R_max, H_set,landmark_states=landmark_states,
                                      delta_s_array=delta_s_array_orig, C_1=C_1, C_2=C_2,
                                      dict_s_opt_actions_arr=dict_s_opt_actions_arr,
                                        state_only_reward_flag=False,  is_delta_s_const=is_delta_s_const, upper_c_for_delt_s=upper_c_for_delt_s, tol=1e-10,
                                        env_name="room", exp_name="landmark_state_design", exp_iter=1, out_file_name="rewards")

    # M_landmark_state_design = (M_orig[0], M_orig[1], R_landmark_state_design, M_orig[3], M_orig[4], M_orig[5])
    # env_landmark_state_design = env_chain.Environment("", M_landmark_state_design)
    env_landmark_state_design = get_env_with_new_R(env_orig, R_landmark_state_design)

    Q_landmark_state_design, V_landmark_state_design, pi_d_landmark_state_design, pi_s_landmark_state_design = \
                            MDPSolver.valueIteration(env_landmark_state_design, env_landmark_state_design.reward, tol=tol)


    # Q_pi, V_pi = MDPSolver.compute_Q_V_Function_given_policy(env_landmark_state_design, pi_target_d,
    #                                                      env_landmark_state_design.reward, tol=tol)
    # print(Q_pi-Q_landmark_state_design)
    # exit(0)


    delta_s_array_landmark_state_design = get_delta_s_given_policy(env_landmark_state_design, pi_target_d, pi_target_s, tol=tol)
    IR_landmark_state_design = calculate_I_R(env_landmark_state_design, pi_d_landmark_state_design, delta_s_array_orig, H_set)


    print("=====================LANDMARK STATE DESIGN=================================")
    print("Landmark_state_design delta_inf = ", min(delta_s_array_landmark_state_design ))
    print("IR_Landmark_state_design  = ", IR_landmark_state_design)
    # print("R_landmark_state_design=\n", R_landmark_state_design)
    print("======================================================")


    # =============== ORIGINAL ====================================

    IR_orig = calculate_I_R(env_orig, pi_d_orig, delta_s_array_orig, H_set)

    print("===================ORIGINAL==================================")
    print("Original delta_inf = ", min(delta_s_array_orig ))
    print("IR_orig = ", IR_orig)
    print("======================================================")


    # ===============ALL STATE DESIGN========================
    s_active_all = copy.deepcopy(range(env_orig.n_states-1))
    _, R_all_state_design = reward_design_model_based(env_orig, pi_target_d, pi_target_s, R_max=R_max, H_set=H_set, s_active=s_active_all,
                                                    delta_s_array=delta_s_array_orig, C_1=C_1, C_2=C_2,
                                                    dict_s_opt_actions_arr=dict_s_opt_actions_arr,
                                                    state_only_reward_flag=state_only_reward_flag,  is_delta_s_const=is_delta_s_const, upper_c_for_delt_s=upper_c_for_delt_s, tol=tol)

    # M_all_state_design = (M_orig[0], M_orig[1], R_all_state_design, M_orig[3], M_orig[4], M_orig[5])
    # env_all_state_design = env_chain.Environment("", M_all_state_design)

    env_all_state_design = get_env_with_new_R(env_orig, R_all_state_design)

    accumulator_for_file["Reward_all_state_design"] = R_all_state_design.flatten()

    Q_all_state_design, V_all_state_design, pi_d_all_state_design, pi_s_all_state_design = \
                            MDPSolver.valueIteration(env_all_state_design, env_all_state_design.reward, tol=tol)
    # print(R_all_state_design)
    # exit(0)


    # Q_pi, V_pi = MDPSolver.compute_Q_V_Function_given_policy(env_landmark_state_design, pi_target_d,
    #                                                      env_landmark_state_design.reward, tol=tol)
    # print(Q_pi-Q_landmark_state_design)
    # exit(0)


    delta_s_array_all_state_design = get_delta_s_given_policy(env_all_state_design, pi_target_d, pi_target_s, tol=tol)
    IR_all_state_design = calculate_I_R(env_all_state_design, pi_d_all_state_design, delta_s_array_orig, H_set)


    print("=====================ALL STATE DESIGN=================================")
    print("ALL_state_design delta_inf = ", min(delta_s_array_all_state_design))
    print("IR_ALL_state_design  = ", IR_all_state_design)
    print("|R_pot - R_all_state_design|_2 = ", np.linalg.norm(R_pot - R_all_state_design))
    print("======================================================")


    # ===============GOAL STATE DESIGN========================
    s_active_goal = env_orig.goal_state
    _, R_goal_state_design = reward_design_model_based(env_orig, pi_target_d, pi_target_s, R_max=R_max, H_set=H_set, s_active=s_active_goal,
                                                    delta_s_array=delta_s_array_orig, C_1=C_1, C_2=C_2,
                                                    dict_s_opt_actions_arr=dict_s_opt_actions_arr,
                                                    state_only_reward_flag=state_only_reward_flag,  is_delta_s_const=is_delta_s_const, upper_c_for_delt_s=upper_c_for_delt_s, tol=tol)

    # M_goal_state_design = (M_orig[0], M_orig[1], R_goal_state_design, M_orig[3], M_orig[4], M_orig[5])
    env_goal_state_design = get_env_with_new_R(env_orig, R_goal_state_design)

    Q_goal_state_design, V_goal_state_design, pi_d_goal_state_design, pi_s_goal_state_design = \
                            MDPSolver.valueIteration(env_goal_state_design, env_goal_state_design.reward, tol=tol)


    # Q_pi, V_pi = MDPSolver.compute_Q_V_Function_given_policy(env_landmark_state_design, pi_target_d,
    #                                                      env_landmark_state_design.reward, tol=tol)
    # print(Q_pi-Q_landmark_state_design)
    # exit(0)


    delta_s_array_goal_state_design = get_delta_s_given_policy(env_goal_state_design, pi_target_d, pi_target_s, tol=tol)
    IR_goal_state_design = calculate_I_R(env_goal_state_design, pi_d_goal_state_design, delta_s_array_orig, H_set)


    print("=====================GOAL STATE DESIGN=================================")
    print("GOAL_state_design delta_inf = ", min(delta_s_array_goal_state_design ))
    print("IR_GOAL_state_design  = ", IR_goal_state_design)
    print("R_or\n", env_orig.reward)
    print("R_goal_design\n", R_goal_state_design)
    print("======================================================")

    write_into_file(accumulator_for_file, exp_iter=1, out_folder_name="designed_rewards_room_const_middle_sub_optimal_IR",
                    out_file_name="information_is_delta_s_const_{}".format(is_delta_s_const))

    exit(0)
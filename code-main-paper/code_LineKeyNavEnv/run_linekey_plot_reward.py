import numpy as np
import copy
import sys
import os
import linekey_MDPSolver
import linekey_plot_heatmap
import argparse
import linekey_reward_design_exprd
import linekey_reward_design_orig
import linekey_reward_design_pbrs
import linekey_utils

import linekey_env
import linekey_abstration

accumulator_for_file = {}



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for Algorithms : name, budget')

    parser.add_argument('--algorithm', nargs='?', const=1, type=str, default="")
    parser.add_argument('--B', nargs='?', const=1, type=int, default=5)
    args = parser.parse_args()
    #Parameters

    dict_s_opt_actions_arr = {}
    n_nonzero_states = args.B + 1
    C_1 = 1
    C_2 = 0 #
    R_max = 10
    state_only_reward_flag = False
    is_delta_s_const = False
    upper_c_for_delt_s = None
    tol = 1e-10

    env_args = {
        "R_max": R_max,
        "gamma": 0.95,
        "terminalState": 1,
        "randomMoveProb": 0.1,
        "epsilon_net": 0.01,
        "n_actions": 3,
    }

    env = linekey_env.Environment(env_args)

    epsilon_net = 1/20

    env_orig = linekey_abstration.discreteEnv(env,
                                            epsilon_net=epsilon_net, num_traj=0, len_traj=500)

    Q_orig, V_orig, pi_d_orig, pi_s_orig = linekey_MDPSolver.valueIteration(env_orig, env_orig.reward, tol=tol)

    # print(get_landmark_reward_for_chain(env_orig, pi_d_orig, 1, R_scalar_for_landmark=2))
    # exit(0)

    #==============================
    pi_target_d = copy.deepcopy(pi_d_orig)
    pi_target_s = copy.deepcopy(pi_s_orig)
    #==============================

    if args.algorithm == "exprd":
        delta_s_array_orig = linekey_utils.get_delta_s_given_policy(env_orig, pi_target_d, pi_target_s, tol)
        s_active = None
        candidate_states = range(0, env_orig.n_states - 1 - 1)
        states_active, R_design, accumulator_for_file = linekey_reward_design_exprd.get_important_states_by_shaping(env_orig, pi_d=pi_target_d, pi_s=pi_target_s,
                                                                  R_max=R_max, H_set=None, s_active=s_active,
                                                                  delta_s_array=delta_s_array_orig, C_1=C_1, C_2=C_2,
                                                                  n_nonzero_states=n_nonzero_states,
                                                                  candidate_states=candidate_states,
                                                                  dict_s_opt_actions_arr=dict_s_opt_actions_arr,
                                                                  state_only_reward_flag=state_only_reward_flag,
                                                                  is_delta_s_const=is_delta_s_const,
                                                                  upper_c_for_delt_s=upper_c_for_delt_s, tol=tol,
                                                                  env_name="room", exp_name="reward_design", exp_iter=1,
                                                                  out_file_name="designed_rewards")

        linekey_plot_heatmap.plotting(env, env_orig, accumulator_for_file, color_ranges=3, B=args.B, name=args.algorithm)
    elif args.algorithm == "pbrs":

        accumulator_for_file = linekey_reward_design_pbrs.get_potential_based_reward(env, env_orig, n_times_state_piced=10,
                        n_times_action_picked=10)

        linekey_plot_heatmap.plotting(env, env_orig, accumulator_for_file, color_ranges=3, B=args.B, name=args.algorithm)

    elif args.algorithm == "orig":

        accumulator_for_file = linekey_reward_design_orig.get_original_reward(env_orig)

        linekey_plot_heatmap.plotting(env, env_orig, accumulator_for_file, color_ranges=3, B=args.B, name=args.algorithm)

    else:
        print("Algorithm names should be chosen from {orig, pbrs, exprd}")
        print("algorithmname: {}".format(args.algorithm))
        exit(0)

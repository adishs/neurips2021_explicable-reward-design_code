import numpy as np
import copy
import sys
import os
import MDPSolver as fourroom_MDPSolver
import fourroom_env as room
import fourroom_plot_heatmap
import argparse
import fourroom_reward_design_exprd
import fourroom_reward_design_orig
import fourroom_reward_design_pbrs
import fourroom_utils as utils

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
        "gridsizefull": 7,
        "R_max": R_max,
        "gamma": 0.95,
        "terminalState": 1,
        "randomMoveProb": 0.1,
    }

    env_orig = room.Environment(env_args)

    Q_orig, V_orig, pi_d_orig, pi_s_orig = fourroom_MDPSolver.valueIteration(env_orig, env_orig.reward, tol=tol)

    # print(get_landmark_reward_for_chain(env_orig, pi_d_orig, 1, R_scalar_for_landmark=2))
    # exit(0)

    #==============================
    pi_target_d = copy.deepcopy(pi_d_orig)
    pi_target_s = copy.deepcopy(pi_s_orig)
    #==============================

    if args.algorithm == "exprd":
        delta_s_array_orig = utils.get_delta_s_given_policy(env_orig, pi_target_d, pi_target_s, tol)
        s_active = None
        candidate_states = range(0, env_orig.n_states - env_orig.terminal_state - 1)
        states_active, R_design, accumulator_for_file = fourroom_reward_design_exprd.get_important_states_by_shaping(env_orig, pi_d=pi_target_d, pi_s=pi_target_s,
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

        fourroom_plot_heatmap.plotting(accumulator_for_file, color_ranges=4, B=args.B, name=args.algorithm)
    elif args.algorithm == "pbrs":

        accumulator_for_file = fourroom_reward_design_pbrs.get_potential_based_reward(env_orig, tol=tol)

        fourroom_plot_heatmap.plotting(accumulator_for_file, color_ranges=4, B=None, name=args.algorithm)

    elif args.algorithm == "orig":

        accumulator_for_file = fourroom_reward_design_orig.get_original_reward(env_orig)

        fourroom_plot_heatmap.plotting(accumulator_for_file, color_ranges=4, B=None, name=args.algorithm)

    else:
        print("Algorithm names should be chosen from {orig, pbrs, exprd}")
        print("algorithmname: {}".format(args.algorithm))
        exit(0)

import copy
import linekey_MDPSolver
import numpy as np
import linekey_utils

accumulator_for_file = {}


def get_potential_based_reward(env_orig, env_abstr, n_times_state_piced=10,
                        n_times_action_picked=10):
    return linekey_utils.calculate_PBRS_orig(env_orig, env_abstr, n_times_state_piced=n_times_state_piced,
                        n_times_action_picked=n_times_action_picked)
    #enddef


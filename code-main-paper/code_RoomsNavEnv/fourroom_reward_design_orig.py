import numpy as np

accumulator_for_file = {}
def get_original_reward(env_orig):
    accumulator_for_file["R_original"] = env_orig.reward.flatten()
    return accumulator_for_file
#enddef


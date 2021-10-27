import numpy as np
import copy
import linekey_MDPSolver
from collections import defaultdict


def evaluate_policy_given_n_episode(env, max_episode=10, policy=None):
    env_copy =copy.deepcopy(env)
    episode_reward_array = []
    for episode in range(max_episode):
        state = env_copy.reset()
        episode_reward = 0
        i = 0

        while True:
            # input()
            action, log_prob = policy(state)
            new_state, reward_orig, done, _ = env_copy.step(action)

            episode_reward += env_copy.gamma**i * reward_orig

            if done:
                episode_reward_array.append(episode_reward)
                break

            state = new_state
            i += 1
    return np.average(episode_reward_array)
# enddef


def evaluate_policy_given_n_episode_q_learning(env, max_episode=10, policy=None):
    env_copy =copy.deepcopy(env)
    episode_reward_array = []
    for episode in range(max_episode):
        state = env_copy.reset()
        if len(state) > 1:
            state = tuple(state)
        else:
            state = state[0]
        # state_int = env.mapping_from_float_coords_to_int_coords[state]
        episode_reward = 0
        i = 0

        while True:
            # input()
            state_int = env.mapping_from_float_coords_to_int_coords[state]
            action = policy[state_int]
            new_state, reward_orig, done, _ = env_copy.step(action)
            if len(new_state) > 1:
                new_state = tuple(new_state)
            else:
                new_state = new_state[0]
            episode_reward += env_copy.gamma**i * reward_orig

            if done:
                episode_reward_array.append(episode_reward)
                break

            state = new_state
            i += 1
    return np.average(episode_reward_array)
# enddef

def get_delta_s_given_policy(env_given, pi_target_d, pi_target_s, tol):

    Q_pi, _ = linekey_MDPSolver.compute_Q_V_Function_given_policy(env_given, pi_target_d,
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

def get_sampling_T_ground(env_orig, n_times_each_state, n_times_each_action):
    state_action_count = defaultdict(float)
    state_action_next_state_count = defaultdict(float)
    reward_count = defaultdict(float)
    accumulator_for_file = {}


    for s_orig in env_orig.states_in_zero_one_interval_with_key[:-1]:
        for i_s in range(n_times_each_state):
            print("===Sampling on original environment===")
            print("s={}".format(s_orig))
            for action in range(env_orig.n_actions):
                for i_a in range(n_times_each_action):
                    env_orig.reset()
                    env_orig.current_state_x = copy.deepcopy(s_orig)
                    env_orig.key = s_orig[1]
                    next_state, reward, done, _ = env_orig.step(action)
                    next_state = tuple(next_state)

                    state_action_count[(s_orig, action)] += 1
                    state_action_next_state_count[(s_orig, action, next_state)] += 1
                    reward_count[(s_orig, action)] += reward
    # exit(0)
    P= {}# = np.zeros((env_orig.n_states, env_orig.n_states, env_orig.n_actions))
    R= {}

    for state in env_orig.states_in_zero_one_interval_with_key:
        for action in range(env_orig.n_actions):
            state_action = (state, action)
            for next_state in env_orig.states_in_zero_one_interval_with_key:
                next_state = tuple(next_state)
                state_action_next_state = (state, action, next_state)
                if state_action_next_state in state_action_next_state_count:
                    P[state, next_state, action] = \
                        state_action_next_state_count[state_action_next_state] / state_action_count[state_action]
                else:
                    P[state, next_state, action] = 0
            if state_action in state_action_count:
                R[state, action] = reward_count[state_action] / state_action_count[state_action]
            else:
                R[state, action] = 0
    return P, R
#enddef

def calculate_PBRS_orig(env_orig, env_abstr, n_times_state_piced=10,
                        n_times_action_picked=10):
    accumulator_for_file = {}

    _, V_abstract, _, _ = linekey_MDPSolver.valueIteration(env_abstr, env_abstr.reward)

    reward_PBRS_int = np.zeros((len(env_orig.states_in_zero_one_interval_with_key), 3))

    T, R = get_sampling_T_ground(env_orig, n_times_state_piced, n_times_action_picked)
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


    accumulator_for_file["R_potential_shaping_cont"] = reward_PBRS_int.flatten()
    return accumulator_for_file
#enddef


def get_transition_chain_key(randomMove=0):
    T_clean = np.zeros((41, 41, 3))

    #state_0
    # T_clean[0, -1, 0] = 1
    T_clean[0, 0, 0] = 1
    T_clean[0, 1, 1] = 0.5
    T_clean[0, 2, 1] = 0.5

    T_clean[0, 0, 2] = 1


    #state_1
    # T_clean[1, -1, 0] = 1
    T_clean[1, 0, 0] = 1
    T_clean[1, 2, 1] = 0.5
    T_clean[1, 3, 1] = 0.5

    T_clean[1, 1, 2] = 1

    #state_2
    T_clean[2, 1, 0] = 0.5
    T_clean[2, 0, 0] = 0.5
    T_clean[2, 3, 1] = 0.5
    T_clean[2, 4, 1] = 0.5

    T_clean[2, 22, 2] = 1 # key picked

    #state_3
    T_clean[3, 2, 0] = 0.5
    T_clean[3, 1, 0] = 0.5
    T_clean[3, 4, 1] = 0.5
    T_clean[3, 5, 1] = 0.5

    T_clean[3, 23, 2] = 1 # key picked

    #state_4
    T_clean[4, 3, 0] = 0.5
    T_clean[4, 2, 0] = 0.5
    T_clean[4, 5, 1] = 0.5
    T_clean[4, 6, 1] = 0.5

    T_clean[4, 4, 2] = 1

    #state_5
    T_clean[5, 4, 0] = 0.5
    T_clean[5, 3, 0] = 0.5
    T_clean[5, 6, 1] = 0.5
    T_clean[5, 7, 1] = 0.5

    T_clean[5, 5, 2] = 1

    #state_6
    T_clean[6, 4, 0] = 0.5
    T_clean[6, 5, 0] = 0.5
    T_clean[6, 7, 1] = 0.5
    T_clean[6, 8, 1] = 0.5

    T_clean[6, 6, 2] = 1

    #state_7
    T_clean[7, 6, 0] = 0.5
    T_clean[7, 5, 0] = 0.5
    T_clean[7, 8, 1] = 0.5
    T_clean[7, 9, 1] = 0.5

    T_clean[7, 7, 2] = 1

    #state_8
    T_clean[8, 7, 0] = 0.5
    T_clean[8, 6, 0] = 0.5
    T_clean[8, 9, 1] = 0.5
    T_clean[8, 10, 1] = 0.5

    T_clean[8, 8, 2] = 1

    #state_9
    T_clean[9, 8, 0] = 0.5
    T_clean[9, 7, 0] = 0.5
    T_clean[9, 10, 1] = 0.5
    T_clean[9, 11, 1] = 0.5

    T_clean[9, 9, 2] = 1

    #state_10
    T_clean[10, 9, 0] = 0.5
    T_clean[10, 8, 0] = 0.5
    T_clean[10, 11, 1] = 0.5
    T_clean[10, 12, 1] = 0.5

    T_clean[10, 10, 2] = 1

    #state_11
    T_clean[11, 10, 0] = 0.5
    T_clean[11, 9, 0] = 0.5
    T_clean[11, 12, 1] = 0.5
    T_clean[11, 13, 1] = 0.5

    T_clean[11, 11, 2] = 1


    #state_12
    T_clean[12, 11, 0] = 0.5
    T_clean[12, 10, 0] = 0.5
    T_clean[12, 13, 1] = 0.5
    T_clean[12, 14, 1] = 0.5

    T_clean[12, 12, 2] = 1

    #state_13
    T_clean[13, 12, 0] = 0.5
    T_clean[13, 11, 0] = 0.5
    T_clean[13, 14, 1] = 0.5
    T_clean[13, 15, 1] = 0.5

    T_clean[13, 13, 2] = 1

    #state_14
    T_clean[14, 13, 0] = 0.5
    T_clean[14, 12, 0] = 0.5
    T_clean[14, 15, 1] = 0.5
    T_clean[14, 16, 1] = 0.5

    T_clean[14, 14, 2] = 1

    #state_15
    T_clean[15, 14, 0] = 0.5
    T_clean[15, 13, 0] = 0.5
    T_clean[15, 16, 1] = 0.5
    T_clean[15, 17, 1] = 0.5

    T_clean[15, 15, 2] = 1

    #state_16
    T_clean[16, 15, 0] = 0.5
    T_clean[16, 14, 0] = 0.5
    T_clean[16, 17, 1] = 0.5
    T_clean[16, 18, 1] = 0.5

    T_clean[16, 16, 2] = 1

    #state_17
    T_clean[17, 16, 0] = 0.5
    T_clean[17, 15, 0] = 0.5
    T_clean[17, 18, 1] = 0.5
    T_clean[17, 19, 1] = 0.5

    T_clean[17, 17, 2] = 1

    #state_18
    T_clean[18, 17, 0] = 0.5
    T_clean[18, 16, 0] = 0.5
    T_clean[18, -1, 1] = 1
    # T_clean[18, 18, 1] = 1

    T_clean[18, 18, 2] = 1

    #state_19
    T_clean[19, 18, 0] = 0.5
    T_clean[19, 17, 0] = 0.5
    T_clean[19, -1, 1] = 1
    # T_clean[19, 19, 1] = 1

    T_clean[19, 19, 2] = 1

    #state_20
    T_clean[20, 20, 0] = 1
    T_clean[20, 21, 1] = 0.5
    T_clean[20, 22, 1] = 0.5

    T_clean[20, 20, 2] = 1


    #state_21
    T_clean[21, 20, 0] = 1
    T_clean[21, 22, 1] = 0.5
    T_clean[21, 23, 1] = 0.5

    T_clean[21, 21, 2] = 1

    #state_22
    T_clean[22, 21, 0] = 0.5
    T_clean[22, 20, 0] = 0.5
    T_clean[22, 23, 1] = 0.5
    T_clean[22, 24, 1] = 0.5

    T_clean[22, 22, 2] = 1

    #state_23
    T_clean[23, 22, 0] = 0.5
    T_clean[23, 21, 0] = 0.5
    T_clean[23, 24, 1] = 0.5
    T_clean[23, 25, 1] = 0.5

    T_clean[23, 23, 2] = 1

    #state_24
    T_clean[24, 23, 0] = 0.5
    T_clean[24, 22, 0] = 0.5
    T_clean[24, 25, 1] = 0.5
    T_clean[24, 26, 1] = 0.5

    T_clean[24, 24, 2] = 1

    #state_25
    T_clean[25, 24, 0] = 0.5
    T_clean[25, 23, 0] = 0.5
    T_clean[25, 26, 1] = 0.5
    T_clean[25, 27, 1] = 0.5

    T_clean[25, 25, 2] = 1

    #state_26
    T_clean[26, 24, 0] = 0.5
    T_clean[26, 25, 0] = 0.5
    T_clean[26, 27, 1] = 0.5
    T_clean[26, 28, 1] = 0.5

    T_clean[26, 26, 2] = 1

    #state_27
    T_clean[27, 26, 0] = 0.5
    T_clean[27, 25, 0] = 0.5
    T_clean[27, 28, 1] = 0.5
    T_clean[27, 29, 1] = 0.5

    T_clean[27, 27, 2] = 1

    #state_28
    T_clean[28, 27, 0] = 0.5
    T_clean[28, 26, 0] = 0.5
    T_clean[28, 29, 1] = 0.5
    T_clean[28, 30, 1] = 0.5

    T_clean[28, 28, 2] = 1

    #state_29
    T_clean[29, 28, 0] = 0.5
    T_clean[29, 27, 0] = 0.5
    T_clean[29, 30, 1] = 0.5
    T_clean[29, 31, 1] = 0.5

    T_clean[29, 29, 2] = 1

    #state_30
    T_clean[30, 29, 0] = 0.5
    T_clean[30, 28, 0] = 0.5
    T_clean[30, 31, 1] = 0.5
    T_clean[30, 32, 1] = 0.5

    T_clean[30, 30, 2] = 1

    #state_31
    T_clean[31, 30, 0] = 0.5
    T_clean[31, 29, 0] = 0.5
    T_clean[31, 32, 1] = 0.5
    T_clean[31, 33, 1] = 0.5

    T_clean[31, 31, 2] = 1


    #state_32
    T_clean[32, 31, 0] = 0.5
    T_clean[32, 30, 0] = 0.5
    T_clean[32, 33, 1] = 0.5
    T_clean[32, 34, 1] = 0.5

    T_clean[32, 32, 2] = 1

    #state_33
    T_clean[33, 32, 0] = 0.5
    T_clean[33, 31, 0] = 0.5
    T_clean[33, 34, 1] = 0.5
    T_clean[33, 35, 1] = 0.5

    T_clean[33, 33, 2] = 1

    #state_34
    T_clean[34, 33, 0] = 0.5
    T_clean[34, 32, 0] = 0.5
    T_clean[34, 35, 1] = 0.5
    T_clean[34, 36, 1] = 0.5

    T_clean[34, 34, 2] = 1

    #state_35
    T_clean[35, 34, 0] = 0.5
    T_clean[35, 33, 0] = 0.5
    T_clean[35, 36, 1] = 0.5
    T_clean[35, 37, 1] = 0.5

    T_clean[35, 35, 2] = 1

    #state_36
    T_clean[36, 35, 0] = 0.5
    T_clean[36, 34, 0] = 0.5
    T_clean[36, 37, 1] = 0.5
    T_clean[36, 38, 1] = 0.5

    T_clean[36, 36, 2] = 1

    #state_37
    T_clean[37, 36, 0] = 0.5
    T_clean[37, 35, 0] = 0.5
    T_clean[37, 38, 1] = 0.5
    T_clean[37, 39, 1] = 0.5

    T_clean[37, 37, 2] = 1


    #state_38
    T_clean[38, 37, 0] = 0.5
    T_clean[38, 36, 0] = 0.5
    T_clean[38, -1, 1] = 1

    T_clean[38, 38, 2] = 1

    #state_39
    T_clean[39, 38, 0] = 0.5
    T_clean[39, 37, 0] = 0.5
    T_clean[39, -1, 1] = 1

    T_clean[39, 39, 2] = 1

    #state_40
    T_clean[-1, -1, :] = 1



    T_final = np.zeros((41, 41, 3))

    for s in range(T_clean.shape[0]):
        #Left
        T_final[s:, :, 0] = (1-randomMove) *T_clean[s, :, 0] +\
            1/2* randomMove*T_clean[s, :, 1] + 1/2*randomMove*T_clean[s, :, 2]

        #RIGHT
        T_final[s:, :, 1] = (1-randomMove) *T_clean[s, :, 1] +\
            1/2* randomMove*T_clean[s, :, 0] + 1/2*randomMove*T_clean[s, :, 2]

        #Pick
        T_final[s:, :, 2] = (1-randomMove) *T_clean[s, :, 2] +\
            1/2* randomMove*T_clean[s, :, 0] + 1/2*randomMove*T_clean[s, :, 1]



        T_final[18, :, 1] = 0
        T_final[18, -1, 1] = 1

        T_final[19, :, 1] = 0
        T_final[19, -1, 1] = 1

        T_final[38, :, 1] = 0
        T_final[38, -1, 1] = 1


        T_final[39, :, 1] = 0
        T_final[39, -1, 1] = 1


    return T_final
#enddef

if __name__ == "__main__":
    pass










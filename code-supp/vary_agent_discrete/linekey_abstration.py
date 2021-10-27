import numpy as np
import copy
import itertools
import random

import linekey_MDPSolver as MDPSolver
from collections import defaultdict
import utils
import linekey_env
import matplotlib.pyplot as plt

import sys


class discreteEnv:
    def __init__(self, env, epsilon_net, num_traj, len_traj):
        self.dimension = 1
        self.env = env
        self.epsilon_net = epsilon_net
        self.num_traj = num_traj
        self.len_traj = len_traj
        self.states_in_zero_one_interval = self.get_states_zero_one_interval()
        self.states_in_zero_one_interval_with_key = self.get_states_zero_one_interval_with_key()

        self.mapping_from_float_coords_to_int_coords = self.get_mapping_from_float_coords_to_int_coords()
        self.n_states = len(list(self.states_in_zero_one_interval_with_key))
        self.gridsizefull = int(np.sqrt(self.n_states))
        self.n_actions = self.env.n_actions
        self.reward = self.get_reward()
        self.T = self.get_transition_matrix()
        self.gamma = env.gamma
        self.InitD = self.get_init_D()
        self.terminal_state = None
        self.goal_state = [39, 38]

    #enddef

    def get_reward_given_cont_state_action(self, cont_state, action):

        int_state = self.mapping_from_con_state_to_int_state(cont_state)

        reward = self.reward[int_state, action]
        return reward
    #enddef

    def get_states_zero_one_interval(self):
        states_zero_one_interval = list(np.round(np.arange(self.epsilon_net / 2, 1, self.epsilon_net), 8))
        states_zero_one_interval.append(-1)
        return states_zero_one_interval
    #enndef

    def get_states_zero_one_interval_with_key(self):
        states_zero_one_interval = list(np.round(np.arange(self.epsilon_net / 2, 1, self.epsilon_net), 8))
        states_zero_one_interval = list(itertools.product(states_zero_one_interval, [0, 1]))
        states_zero_one_interval = sorted(states_zero_one_interval, key=lambda tup: tup[1])
        states_zero_one_interval.append((-1, 0))
        return states_zero_one_interval
    # enndef

    def mapping_from_con_state_to_int_state(self, coord_x):
        nearest = self.phi(coord_x)
        return self.mapping_from_float_coords_to_int_coords[nearest]
    #enndef

    def phi(self, coord_x):
        return self.find_nearest(coord_x)
    #enddef

    def find_nearest(self, coord_x):
        array = np.asarray(self.states_in_zero_one_interval_with_key)
        diff_array = array - [coord_x[0], coord_x[1]]
        tmp_array = np.round([np.sqrt(i ** 2 + j ** 2) for i, j in diff_array], 10)
        # print(tmp_array)
        idx = np.array(tmp_array).argmin()
        min_elment = (array[idx][0], array[idx][1])
        return min_elment
    # enddef

    def get_mapping_from_float_coords_to_int_coords(self):

        mapping = dict(
            zip(        self.states_in_zero_one_interval_with_key,
                        range(0, len(self.states_in_zero_one_interval_with_key)),
                    ))
        return mapping
    #enddef

    def get_reward(self):
        reward = np.zeros((self.n_states, self.n_actions))

        reward[38, 1] = 10
        reward[39, 1] = 10

        return reward
    #enddef

    def get_init_D(self):
        InitD = 0 * np.ones(self.n_states)
        # InitD /= sum(InitD)
        InitD[10] = 0.5
        InitD[11] = 0.5
        return InitD
    # enddef

    def get_next_state(self, s_t, a_t):
        next_state = np.random.choice(np.arange(0, self.n_states, dtype="int"), size=1, p=self.T[s_t, :, a_t])[0]
        return next_state
    #enddef

    def get_transition_matrix(self):
        T = utils.get_transition_chain_key(self.env.randomMoveProb)
        return T
    #enddef


#endclass


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


def get_sampling(env_cont, env_discrete, n_times_each_state, n_times_each_action):
    state_action_count = defaultdict(float)
    state_action_next_state_count = defaultdict(float)


    for s in range(env_disc.n_states-1):
        for i_s in range(n_times_each_state):
            for action in range(env_disc.n_actions):
                for i_a in range(n_times_each_action):
                    env_cont.reset()
                    coord_x = env_discrete.states_in_zero_one_interval_with_key[s]
                    env_cont.current_state_x = copy.deepcopy(coord_x)
                    env_cont.key = coord_x[1]
                    next_state, reward, done, _ = env_cont.step(action)

                    # s_int = env_discrete.mapping_from_con_state_to_int_state(state)
                    s_next_int = env_discrete.mapping_from_con_state_to_int_state(next_state)
                    print("s={}, a={}, s_n={}".format(s, action, s_next_int))
                    # if s == 3:
                    #     input()

                    state_action_count[(s, action)] += 1
                    state_action_next_state_count[(s, action, s_next_int)] += 1

    P = np.zeros((env_discrete.n_states, env_discrete.n_states, env_discrete.n_actions))

    for state in range(env_discrete.n_states):
        for action in range(env_discrete.n_actions):
            for next_state in range(env_discrete.n_states):
                state_action = (state, action)
                state_action_next_state = (state, action, next_state)
                if state_action_next_state in state_action_next_state_count:
                    # print("==============")
                    # print("(state, action, next_state)", state_action_next_state)
                    # print("state_action_next_state_count=", state_action_next_state_count[state_action_next_state])
                    # print("state_action_count=",  state_action_count[state_action])
                    # input()
                    P[state, next_state, action] = \
                        state_action_next_state_count[state_action_next_state] / state_action_count[state_action]
    return P
#enddef

def get_sampling_T_ground(env_orig, n_times_each_state, n_times_each_action):
    state_action_count = defaultdict(float)
    state_action_next_state_count = defaultdict(float)
    reward_count = defaultdict(float)


    for s_orig in env_orig.states_in_zero_one_interval_with_key[:-1]:
        for i_s in range(n_times_each_state):
            for action in range(env_orig.n_actions):
                for i_a in range(n_times_each_action):
                    env_orig.reset()
                    env_orig.current_state_x = copy.deepcopy(s_orig)
                    env_orig.key = s_orig[1]
                    next_state, reward, done, _ = env_orig.step(action)
                    next_state = tuple(next_state)

                    # s_int = env_orig.mapping_from_con_state_to_int_state(s_orig)
                    # s_next_int = env_orig.mapping_from_con_state_to_int_state(next_state)
                    print("s={}, a={}, s_n={}".format(s_orig, action, next_state))
                    # if s == 3:
                    #     input()

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
                    # print("==============")
                    # print("(state, action, next_state)", state_action_next_state)
                    # print("state_action_next_state_count=", state_action_next_state_count[state_action_next_state])
                    # print("state_action_count=",  state_action_count[state_action])
                    # input()
                    P[state, next_state, action] = \
                        state_action_next_state_count[state_action_next_state] / state_action_count[state_action]
                else:
                    P[state, next_state, action] = 0
            if state_action in state_action_count:
                R[state, action] = reward_count[state_action] / state_action_count[state_action]
            else:
                R[state, action] = 0
    # P[(env_orig.states_in_zero_one_interval_with_key[-1],
    #   env_orig.states_in_zero_one_interval_with_key[-1], 0)] = 1
    # P[(env_orig.states_in_zero_one_interval_with_key[-1],
    #   env_orig.states_in_zero_one_interval_with_key[-1], 1)] = 1
    # P[(env_orig.states_in_zero_one_interval_with_key[-1],
    #   env_orig.states_in_zero_one_interval_with_key[-1], 2)] = 1

    return P, R
#enddef

def get_env_with_new_R(env_orig, reward):
    env_new = copy.deepcopy(env_orig)
    env_new.reward = copy.deepcopy(reward)
    return env_new
# ednndef
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
def mapping_rewards_rom_abstraction_to_orig_env(env_orig, env_abstr):

    reward = {}
    env_with_anstracted_reward = copy.deepcopy(env_orig)
    for s_orig in env_orig.states_in_zero_one_interval_with_key:
        s_abtracted_int = env_abstr.mapping_from_con_state_to_int_state(s_orig)
        for a in range(env_orig.n_actions):
            reward[s_orig, a] = env_abstr.reward[s_abtracted_int, a]
    env_with_anstracted_reward.reward = reward
    return env_with_anstracted_reward
#enddef

def calculate_PBRS_orig(env_orig, env_abstr, n_times_state_piced=10,
                        n_times_action_picked=10):

    _, V_abstract, _, _ = MDPSolver.valueIteration(env_abstr, env_abstr.reward)
    reward_PBRS = {}

    T, R = get_sampling_T_ground(env_orig, n_times_state_piced, n_times_action_picked)

    for state in env_orig.states_in_zero_one_interval_with_key:
        for action in range(env_orig.n_actions):
            sum_over_next_state = 0
            for next_state in env_orig.states_in_zero_one_interval_with_key:
                next_state = tuple(next_state)
                sum_over_next_state += env_orig.gamma * T[state, next_state, action] * \
                    V_abstract[env_abstr.mapping_from_con_state_to_int_state(next_state)]
            reward_PBRS[state, action] = R[state, action] + sum_over_next_state - \
                    V_abstract[env_abstr.mapping_from_con_state_to_int_state(state)]

    return reward_PBRS
#enddef



#enddef
if __name__ == "__main__":
    pass
import numpy as np
import copy
import itertools
import random

import linekey_MDPSolver as MDPSolver
from collections import defaultdict

import sys


class discreteEnv:
    def __init__(self, env_args):
        self.dimension = 2
        self.epsilon_net = env_args["epsilon_net"]
        self.randomMoveProb = env_args["randomMoveProb"]
        self.R_max = env_args["R_max"]
        self.n_actions = env_args["n_actions"]
        self.actions = np.array([0, 1, 2])
        self.gamma = env_args["gamma"]
        self.terminal_state = env_args["terminalState"]
        self.delta_move = [0.074, 0.076]
        self.reward_range_x = [0.9, 1]
        self.small_noise = [0, 0]
        self.s_0_state_range = [0, 0.1]

        self.states_in_zero_one_interval = self.get_states_zero_one_interval()
        self.states_in_zero_one_interval_with_key = self.get_states_zero_one_interval_with_key()
        self.n_states = len(self.states_in_zero_one_interval_with_key)
        self.mapping_from_float_coords_to_int_coords = self.get_mapping_from_float_coords_to_int_coords()
        self.mapping_from_int_coords_to_float_coords = self.get_mapping_from_int_coords_to_float_coords()
        # self.reward = self.get_reward()
        # self.T = self.get_transition_matrix_line()
        # self.InitD = self.get_init_D()

        self.current_state_x = None
        self.key = 0
        self.start_state = self.find_nearest((0.5, 0))
        self.key_location = [self.find_nearest((0.1, 0)), self.find_nearest((0.2, 0))]
        self.H = 50
        self.steps = 0
        self.done = False
        self.reward = None

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


    def transitions_LEFT(self):
        move_size_x = - random.uniform(self.delta_move[0], self.delta_move[1]) +\
                      random.uniform(self.small_noise[0], self.small_noise[1])
        if self.current_state_x[0] + move_size_x <= 0:
            self.current_state_x = self.find_nearest((0, self.key))
        else:
            next_state = (self.current_state_x[0] + move_size_x, self.key)
            self.current_state_x = self.find_nearest(next_state)

        return self.current_state_x
    #enddef

    def transitions_RIGHT(self):
        move_size_x = random.uniform(self.delta_move[0], self.delta_move[1]) +\
                      random.uniform(self.small_noise[0], self.small_noise[1])
        if self.current_state_x[0] + move_size_x >= 1:
            self.current_state_x = self.find_nearest((1, self.key))
        else:
            next_state = (self.current_state_x[0] + move_size_x, self.key)
            self.current_state_x = self.find_nearest(next_state)

        return self.current_state_x
    #enddef

    def pick_KEY(self):

        if self.key_location[0][0] <= self.current_state_x[0] <= self.key_location[1][0]:
            self.key = 1
            self.current_state_x = self.find_nearest((self.current_state_x[0], 1))
        return self.current_state_x
    #enddef

    def get_random_action_coords(self, action):
        random_actions = np.setdiff1d(self.actions, action)

        random_action_chosen = np.random.choice(random_actions)

        if random_action_chosen == 0:  # UP
            next_x = self.transitions_LEFT()

        elif random_action_chosen == 1: #LEFT
            next_x = self.transitions_RIGHT()

        elif random_action_chosen == 2: #DOWN
            next_x = self.pick_KEY()
        return next_x
    #enndef


    def reset(self):
        self.current_state_x = self.start_state
        self.steps = 0
        self.done = False
        self.key = 0
        return np.array(self.current_state_x)
    #enddef

    def step(self, action):
        reward = self.get_reward(action)
        next_x = self.get_transition(action)

        if self.steps > self.H or self.current_state_x[0] == -1:
            self.done = True

        self.current_state_x = copy.deepcopy(next_x)

        next_state = next_x
        self.steps += 1

        return np.array(next_state), reward, self.done, None
    #enddef

    def get_reward(self, action):

        if self.reward_range_x[0] <= self.current_state_x[0] <= self.reward_range_x[1] \
                and action == 1 and self.key==1:
            return self.R_max #return reward

        elif (0 <= self.current_state_x[0] <= 1):
            return 0

        else:
            print("x is out of the range [0, 1]"
                  .format(self.current_state_x))
            exit(0)
    #enddef

    def get_transition(self, action):
        if self.reward_range_x[0] <= self.current_state_x[0] < self.reward_range_x[1] \
                and (action == 1): #Right action terminate goal
            self.current_state_x = (-1, 0)
            return (-1, 0)

        # elif self.s_0_state_range[0] <= self.current_state_x[0] <= self.s_0_state_range[1] and \
        #          (action == 0):  # left action terminate from 0 state
        #     self.current_state_x = (-1, 0)
        #     return (-1, 0)

        elif action == 0:  # LEFT action
            random_number = np.random.random()
            if random_number >= self.randomMoveProb:  # with prob 1-randomMoveProb go LEFT
                next_x = self.transitions_LEFT()
            else:  # with prob 0.1 go any other direction
                next_x = self.get_random_action_coords(action)

        elif action == 1:  # RIGHT action
            random_number = np.random.random()
            if random_number >= self.randomMoveProb:  # with prob 1-randomMoveProb go RIGHT
                next_x = self.transitions_RIGHT()
            else:  # with prob 0.1 go any other direction
                next_x = self.get_random_action_coords(action)

        elif action == 2:  # PICK action
            random_number = np.random.random()
            if random_number >= self.randomMoveProb:  # with prob 1-randomMoveProb go RIGHT
                next_x = self.pick_KEY()
            else:  # with prob 0.1 go any other direction
                next_x = self.get_random_action_coords(action)
        else:
            print(action)
            exit(0)

        return next_x
    #enddef



    def get_reward_given_cont_state_action(self, cont_state, action):

        int_state = self.mapping_from_con_state_to_int_state(cont_state)

        reward = self.reward[int_state, action]
        return reward
    #enddef

    def get_states_zero_one_interval(self):
        states_zero_one_interval = list(np.round(np.arange(self.epsilon_net / 2, 1, self.epsilon_net), 8))
        # states_zero_one_interval = list(itertools.product(states_zero_one_interval, [0, 1]))
        states_zero_one_interval.append(-1)
        # print(states_zero_one_interval)
        # exit(0)
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


    def get_mapping_from_int_coords_to_float_coords(self):
        mapping = dict(
            zip(        range(0, len(self.states_in_zero_one_interval)),
                        self.states_in_zero_one_interval
                    ))
        return mapping
    #enndef

    def get_mapping_from_float_coords_to_int_coords(self):

        mapping = dict(
            zip(        self.states_in_zero_one_interval_with_key,
                        range(0, len(self.states_in_zero_one_interval_with_key)),
                    ))
        return mapping
    #enddef

    def get_init_D(self):
        InitD = 0 * np.ones(self.n_states)
        # InitD /= sum(InitD)
        InitD[0] = 1
        return InitD
    # enddef

    def get_next_state(self, s_t, a_t):
        next_state = np.random.choice(np.arange(0, self.n_states, dtype="int"), size=1, p=self.T[s_t, :, a_t])[0]
        return next_state
    #enddef

    # def get_transition_matrix(self):
    #     T = utils.get_transition_chain(self.env.randomMoveProb)
    #     return T
    # #enddef


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


    for s in range(20):
        for i_s in range(n_times_each_state):
            for action in range(2):
                for i_a in range(n_times_each_action):
                    env_cont.reset()
                    coord_x = env_discrete.states_in_zero_one_interval[s]
                    env_cont.current_state_x = copy.deepcopy(coord_x)
                    next_state, reward, done, _ = env_cont.step(action)

                    # s_int = env_discrete.mapping_from_con_state_to_int_state(state)
                    s_next_int = env_discrete.mapping_from_con_state_to_int_state(next_state)
                    print("s={}, a={}, s_n={}".format(s, action, s_next_int))
                    # if s == 74:
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

def get_env_with_new_R(env_orig, reward):
    env_new = copy.deepcopy(env_orig)
    env_new.reward = copy.deepcopy(reward)
    return env_new
# ednndef
#enddef
if __name__ == "__main__":
    pass
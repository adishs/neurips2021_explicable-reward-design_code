import numpy as np
import copy
import itertools
import random
import sys
sys.path.append('../')
import linekey_MDPSolver as MDPSolver
from collections import defaultdict
import utils
import linekey_env_cont
import matplotlib.pyplot as plt
import reinforce



from reward_design.reward_design_optimization_model_sub_optimal_IR import *


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
                    state_action = (env_orig.find_nearest(s_orig), action)
                    state_action_nex_state = (env_orig.find_nearest(s_orig), action,
                                              env_orig.find_nearest(next_state))
                    state_action_count[state_action] += 1
                    state_action_next_state_count[state_action_nex_state] += 1
                    reward_count[state_action] += reward
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


    env_args = {
        "R_max": 10,
        "gamma": 0.95,
        "terminalState": 1,
        "randomMoveProb": 0.1,
        "epsilon_net": 0.01,
        "n_actions": 3,
    }
    epsilon_net = 1/20

    env = env_chain_discrete_0_1_key.discreteEnv(env_args)

    env_disc = discreteEnv(env, epsilon_net=epsilon_net, num_traj=0, len_traj=500)
    # Reward_PBRS = calculate_PBRS_orig(env, env_disc)
    # env_orig_abstracted_reward = copy.deepcopy(env)
    # env_orig_abstracted_reward.reward = Reward_PBRS
    # q_learning_step.q_learning(env_orig_abstracted_reward, max_episode=50000,
    #                            max_step=500, epsilon=0.1, alpha=0.5, env_orig=env,
    #                             pi_t=None, name="", usereward_env=True)

    # print(env_disc.states_in_zero_one_interval_with_key[37])
    # exit(0)

    # T_sampling = get_sampling(env, env_disc, n_times_each_state=30, n_times_each_action=30)
    # array_fo = []
    # for s in range(env_disc.n_states - 1):
    #     for a in range(env_disc.n_actions):
    #         x = np.max(abs(T_sampling[s, :, a] - env_disc.T[s, :, a]))
    #         print("max |T_samp[{},:,{}]-T_hand[{}, :, {}]|".format(s, a, s, a),
    #               x)
    #         array_fo.append(x)
    # print(np.round(sorted(array_fo), 4))
    # exit(0)

    print(len(env_disc.states_in_zero_one_interval_with_key))
    Q_disc, V_disc, pi_d_disc, pi_s_disc = MDPSolver.valueIteration(env_disc, env_disc.reward)
    print(np.round(V_disc, 4))

    # env_tmp = copy.deepcopy(env_disc)
    # R_pot = get_potential_based_reward(env_disc)
    # env_tmp.reward = R_pot
    #
    # env_orig_abstracted_reward = mapping_rewards_rom_abstraction_to_orig_env(env, env_tmp)
    #
    # print(env_orig_abstracted_reward.reward)
    # print(R_pot[18])
    # print(R_pot[19])



    # exit(0)

    # calculate_PBRS_orig
    # exit(0)
    # print(pi_d_disc)
    # # input()
    # # exit(0)
    # print(calculate_PBRS_orig(env, env_disc, V_disc))
    # exit(0)




    # Q_disc, V_disc, pi_d_disc, pi_s_disc = MDPSolver.valueIteration(env_disc, env_disc.reward)
    # print(pi_d_disc)

    H_set = [1, 2, 4, 8, 16, 32]

    delta_s_array = get_delta_s_given_policy(env_disc, pi_d_disc, pi_s_disc, tol=1e-10)
    upper_c_for_delt_s = sorted(delta_s_array)[len(delta_s_array) // 2]

    # print(env_net.n_actions)
    # print(env_net.n_states)
    # exit(0)



    # s_active = [39, 38, 2, 3, 10, 5, 15, 20, 25, 30, 35] #fails
    s_active = [39, 38, 3, 2, 20, 6, 9]
    # s_active = [19, 18, 0, 1]
    # s_active = range(0, 40)

    _, R_sol = reward_design_model_based(env_disc, pi_d=pi_d_disc, pi_s=pi_s_disc, R_max=10,
                                         H_set=H_set, s_active=s_active,
                              delta_s_array=delta_s_array, C_1=1, C_2=0, dict_s_opt_actions_arr={},
                              state_only_reward_flag=False, is_delta_s_const=True,
                              upper_c_for_delt_s=upper_c_for_delt_s, tol=1e-10)
    print(R_sol)
    # exit(0)
    R_pot = get_potential_based_reward(env_disc)
    discrete_env_shaped = get_env_with_new_R(env_disc, R_sol)
    Q_sol, V_sol, pi_d_sol, pi_s_sol = MDPSolver.valueIteration(discrete_env_shaped, discrete_env_shaped.reward)

    print(pi_d_sol-pi_d_disc)
    print("=================")
    print("=================")
    # input()
    print("discrete_env_shaped.reward")
    print(discrete_env_shaped.reward)
    input()

    env_orig_abstracted_reward = mapping_rewards_rom_abstraction_to_orig_env(env, discrete_env_shaped)

    agent = reinforce.Agent(env_orig_abstracted_reward, usereward_env=True, learning_rate=3e-4)
    agent.train(5000, 500)


    # q_learning_step.q_learning(env_orig_abstracted_reward, max_episode=50000,
    #                            max_step=500, epsilon=0.1, alpha=0.5, env_orig=env,
    #                             pi_t=None, name="", usereward_env=True)
    # exit(0)
    # result_dict = q_learning.q_learning(discrete_env_shaped, max_episode=3000, max_step=500,
    #                       epsilon=0.1, alpha=0.5, env_orig=env_disc,
    #            pi_t=None, name="", tol=1e-5)
    #
    # res_1 = None
    # res_1_key = None
    # res_2=None
    # res_2_key = None
    # for key in result_dict:
    #     res = result_dict[key]
    #     print(key)
    #
    #     plt.plot(np.array(res)[::200], label=str(key))
    # plt.legend()
    # plt.show()
    # exit(0)
    #
    #
    exit(0)
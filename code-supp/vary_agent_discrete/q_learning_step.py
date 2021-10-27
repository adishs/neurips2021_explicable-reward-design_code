import numpy as np
import copy
import linekey_MDPSolver as MDPSolver
import utils

import time

import matplotlib.pyplot as plt


def make_epsilon_greedy_policy(Q, state, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    A[best_action] += (1.0 - epsilon)

    return A
#enddef


def q_learning(env, max_episode=3000, max_step=500, epsilon=0.1, alpha=0.5, env_orig=None,
               pi_t=None, name="", usereward_env=False, tol=1e-5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        Q is the optimal action-value function, a dictionary mapping state -> action values.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = np.zeros((env_orig.n_states, env_orig.n_actions))

    result_dict = {}
    result_dict[name+"_expected_reward"] = []
    # result_dict[name+"_steps_to_reach_goal"] = []




    for i_episode in range(max_episode + 1):
        s_a_pairs = []
        # Print out which episode we're on, useful for debugging.
        # result_dict[name+"_expected_reward"].append(np.dot(V_tmp, original_env.InitD))

        # Reset the environment and pick the first state
        state = env_orig.reset()
        if len(state) > 1:
            state = tuple(state)
        else:
            state=state[0]

        pi_d = make_greedy_det_policy(Q)

        exp_reward = utils.evaluate_policy_given_n_episode_q_learning(env_orig,
                                                                      max_episode=10, policy=pi_d)
        if (i_episode) % 100 == 0:
            print("============{}====================".format(name))
            print("\rEpisode {}/{}.".format(i_episode + 1, max_episode), end="")
            print("expected_reward=", exp_reward)
            # print(np.argmax(Q, axis=1))

        result_dict[name + "_expected_reward"].append(exp_reward)
        for step in range(max_step+1):
            state_int = env_orig.mapping_from_float_coords_to_int_coords[state]
            # Take a step
            action_probs = make_epsilon_greedy_policy(Q, state_int, epsilon, env.n_actions)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, r, done, _ = env_orig.step(action)
            if len(next_state) > 1:
                next_state = tuple(next_state)
            else:
                next_state = next_state[0]
            # next_state=tuple(next_state)
            # next_state = env.get_next_state(state, action)
            next_state_int = env_orig.mapping_from_float_coords_to_int_coords[next_state]

            # s_a_pairs.append((state, action, next_state))

            # r = env.reward[state, action]
            if usereward_env:
                r = env.reward[state, action]
                # print("state={}, action={}, reward={}".format(
                #     env.mapping_from_con_state_to_int_state(state), action, r))
                # if (env.mapping_from_con_state_to_int_state(state) == 38 or \
                #         env.mapping_from_con_state_to_int_state(state)==39) and action == 1:
                #     print("episode=", i_episode)
                #     print("step=", step)
                #     print("orig_state", state)
                #     print(r_orig)
                #     input()

            # TD Update
            best_next_action = np.argmax(Q[next_state_int])
            td_target = r + env.gamma * Q[next_state_int][best_next_action]
            td_delta = td_target - Q[state_int][action]
            Q[state_int][action] = Q[state_int][action] + alpha * td_delta


            # if state in env.goal_state:
            #     if not visit_goal_flag:
            #         visited_goal_step = copy.deepcopy(step)
            #         visit_goal_flag = True

            if done: #if we visit  a terminal state break
                break

            # if env.terminal_state == 0: #if no terminal state  and we visited goal
            #     if state in env.goal_state:
            #         break

            state = copy.deepcopy(next_state)
        # result_dict[name + "_steps_to_reach_goal"].append(visited_goal_step)

    return result_dict
# enddef



def q_learning_conv_det_policy(env, max_episode=3000, max_step=500, epsilon=0.1, alpha=0.5,
               name = "", pi_t=None, tol=1e-5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        Q is the optimal action-value function, a dictionary mapping state -> action values.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = np.zeros((env.n_states, env.n_actions))

    steps_to_reach_goal = []
    steps_to_converge_to_target_pi_array = []
    steps_to_converge_to_target_pi = None


    for i_episode in range(max_episode + 1):
        s_a_pairs = []
        # Print out which episode we're on, useful for debugging.
        # result_dict[name+"_expected_reward"].append(np.dot(V_tmp, original_env.InitD))

        if (i_episode) % 10 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, max_episode), end="")

        # Reset the environment and pick the first state
        state = np.random.choice(np.arange(env.n_states), p=env.InitD)


        for step in range(max_step+1):
            # Take a step
            action_probs = make_epsilon_greedy_policy(Q, state, epsilon, env.n_actions)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state = env.get_next_state(state, action)

            s_a_pairs.append((state, action, next_state))
            r = env.reward[state, action]
            # print("s={}, a={} r(s)={}".format(state, action, r))

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = r + env.gamma * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alpha * td_delta

            pi_greedy_from_Q = make_greedy_det_policy(Q)
            # print("=============")
            # print(Q)
            # print("=============")
            # input()
            if (pi_t == pi_greedy_from_Q).all():
                steps_to_converge_to_target_pi_array.append(i_episode*max_step+step)
                steps_to_converge_to_target_pi = i_episode*max_step+step
                return Q, steps_to_converge_to_target_pi
            state = copy.deepcopy(next_state)

            if env.terminal_state == 1 and state == env.n_states-1:
                break



        #     if state in env.goal_states:
        #         # if len(s_a_pairs)< 15:
        #         #     print(s_a_pairs)
        #         #     print(env.goal_states)
        #         #     exit(0)
        #         break
        #     state = copy.deepcopy(next_state)
        # steps_to_reach_goal.append(step)
        # # print(t)


    return Q, max_step*max_episode
# enddef


def get_R_attacked_shaped(env, V):
    R_attacked_shaped = np.zeros((env.n_states, env.n_states, env.n_actions))

    for s in range(env.n_states):
        for s_n in range(env.n_states):
            for a in range(env.n_actions):
                R_attacked_shaped[s, s_n, a] = env.reward[s,a] + F(env, s, a, s_n, V)
    return R_attacked_shaped
#enddef

def F(env, s, a, s_n, V):
    return env.gamma * V[s_n] - V[s]
#enddef

def get_states_given_cell(cell, phi):
    states = []
    for key in phi:
        if phi[key] == cell:
            states.append(key)
    return states
    #enddef

# def my_argmax(array):
#     actions_array = np.where(array == array.max())[0]
#     action = np.random.choice(actions_array, size=1)[0]
#     return action
# #enddef


def compute_policy(Q):
    # calculate Value Function
    V = np.max(Q, axis=1)

    # DETERMINISTIC
    pi_d = np.argmax(Q, axis=1)
    # Stochastic
    pi_s = Q - np.max(Q, axis=1)[:, None]
    pi_s[np.where((-1e-2 <= pi_s) & (pi_s <= 1e-2))] = 1
    pi_s[np.where(pi_s <= 0)] = 0
    pi_s = pi_s / pi_s.sum(axis=1)[:, None]

    return Q, V, pi_d, pi_s
# enddef

def make_greedy_det_policy(Q):
    # DETERMINISTIC
    pi_d = np.argmax(Q, axis=1)
    return pi_d
#enddef

if __name__ == "__main__":

    env_name = "eps=0.5_corrected_shaping_chain_8_prob=0.9"
    env_argv = {"n_states": 15,
                "n_actions": 2,
                "randomMoveProb": 0.01,
                "gamma": 0.99
                }
    env = env_chain.Environment(env_argv)
    print(env.reward)
    Q, acc_0 = q_learning(env, max_episode=50000, max_step=200, epsilon=0.1, alpha=0.5, V=None, use_shaping=False,
              env_M_bar=None, use_M_bar=False, only_reward_change=False)
    _, _, pi_d_Q, pi_s = compute_policy(Q)

    # Q_shaping, acc_0_shaping = q_learning(env, num_episodes=1000, num_iter=300,
    #                           discount_factor=0.99, alpha=0.5, epsilon=0.1,
    #                           name="M_shaping", env_orig=env, shaping=True)
    # _, _, pi_d_Q_shaping, _ = compute_policy(Q_shaping)

    _, V, pi_opt, _ = MDPSolver.valueIteration(env, env.reward)
    print()
    print("pi_d_Q=\n",pi_d_Q)

    # print("pi_d_Q_shaping=", pi_d_Q_shaping)
    print("============================")






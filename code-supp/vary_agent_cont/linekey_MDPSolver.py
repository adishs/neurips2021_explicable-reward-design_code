
import numpy as np
import copy
import numpy.matlib
import random
import time
from scipy import sparse
import matplotlib.pyplot as plt
import time
np.set_printoptions(suppress=True);
np.set_printoptions(precision=12);
# np.set_printoptions(threshold=np.nan);
np.set_printoptions(linewidth=500)


# def get_stationary_dist(env, policy, tol=1e-3):
#     start = time.time()
#     if policy.ndim == 1:
#         policy = convert_det_to_stochastic_policy(env, policy)
#     P_pi_sparse, P_pi = get_P_pi_sparse_transpose(env, policy)
#     d_pi = copy.deepcopy(env.InitD)
#     mixing_time = 0
#     while True:
#         mixing_time += 1
#         d_pi_old = copy.deepcopy(d_pi)
#         d_pi[:] = P_pi_sparse.dot(d_pi_old)
#         if (np.linalg.norm(d_pi - d_pi_old, np.inf) < tol):
#             break
#     #print("Stationary Distribution\n", d_pi)
#     print("end_iter", time.time()-start)
#     return d_pi
# # enddef

# def get_stationary_dist(env, policy):
#     start = time.time()
#     n_states = env.n_states
#     P_pi = np.zeros((n_states, n_states))
#     for s in range(n_states):
#         for s_n in range(n_states):
#             P_pi[s, s_n] = env.T[s, s_n, policy[s]]
#     A = np.zeros((n_states, n_states))
#     A[0: n_states - 1, :] = np.transpose(P_pi - np.identity(n_states))[1:, :]
#     A[-1, :] = np.ones(n_states)
#     b = np.zeros(n_states)
#     b[-1] = 1
#     mu = np.linalg.solve(A, b)
#     print("Time to calculate state dist using lp:", time.time()-start)
#     return mu
# #enddef

def get_stationary_dist(env, policy):
    start = time.time()
    n_states = env.n_states
    P_pi = get_P_pi(env, policy)
    A = np.zeros((n_states, n_states))
    A[0: n_states - 1, :] = np.transpose(P_pi - np.identity(n_states))[1:, :]
    A[-1, :] = np.ones(n_states)
    b = np.zeros(n_states)
    b[-1] = 1
    mu = np.linalg.solve(A, b)
    # print("Time to calculate state dist using lp:", time.time()-start)
    return mu
#enddef

def get_P_pi_sparse_transpose(env, policy):
    if policy.ndim == 1:
        policy = convert_det_to_stochastic_policy(env, policy)
    P_pi = np.zeros((env.n_states, env.n_states))
    for n_s in range(env.n_states):
        for a in range(env.n_actions):
            P_pi[:, n_s] += policy[:, a] * env.T[:, n_s, a]

    P_pi = np.transpose(P_pi)
    P_pi_sparse = sparse.csr_matrix(P_pi)
    return P_pi_sparse, P_pi
# enddef

def get_P_pi_sparse(env, policy):
    P_pi = np.zeros((env.n_states, env.n_states))
    for n_s in range(env.n_states):
        for a in range(env.n_actions):
            P_pi[:, n_s] += policy[:, a] * env.T[:, n_s, a]

    P_pi_sparse = sparse.csr_matrix(P_pi)
    return P_pi_sparse, P_pi
# enddef

def valueIteration_number_iter(env, reward, tol=1e-6):
    #print("inside valueIteration")
    V = np.zeros((env.n_states))
    Q = np.zeros((env.n_states, env.n_actions))
    #print("reward",reward)
    #print("V shape=", V.shape, "V type=", type(V))
    #print("Q shape=", Q.shape, "Q type=", type(Q))
    #print("reward shape=", reward.shape, "reward type=", type(reward))
    iter=0
    while True:
        iter +=1
        V_old = copy.deepcopy(V)
        #print("before for loop")
        for a in range(env.n_actions):
            Q[:, a] = reward[:, a] + env.gamma * env.T[:,:,a].dot(V)
        #print("after for loop")
        V = np.max(Q, axis=1)
        #print("iter=", iter, " tol=", abs(np.linalg.norm(V - V_old, np.inf)))
        if abs(np.linalg.norm(V - V_old, np.inf)) < tol:
            break
    #endwhile

    return iter
#enddef


def valueIteration(env, reward, tol=1e-6):
    #print("inside valueIteration")
    V = np.zeros((env.n_states))
    Q = np.zeros((env.n_states, env.n_actions))
    #print("reward",reward)
    #print("V shape=", V.shape, "V type=", type(V))
    #print("Q shape=", Q.shape, "Q type=", type(Q))
    #print("reward shape=", reward.shape, "reward type=", type(reward))
    iter=0
    while True:
        iter +=1
        V_old = copy.deepcopy(V)
        #print("before for loop")
        for a in range(env.n_actions):
            Q[:, a] = reward[:, a] + env.gamma * env.T[:,:,a].dot(V)
        #print("after for loop")
        V = np.max(Q, axis=1)
        #print("iter=", iter, " tol=", abs(np.linalg.norm(V - V_old, np.inf)))
        if abs(np.linalg.norm(V - V_old, np.inf)) < tol:
            break
    #endwhile

    # For a deterministic policy
    pi_d = np.argmax(Q, axis=1)
    #print("pi_d=", pi_d)
    #print("pi_d shape=", pi_d.shape, "pi_d type=", type(pi_d), "pi_d[0] type=", type(pi_d[0]))
    #print("Q=\n", Q)
    # For a non-deterministic policy
    #pi_s = np.zeros((env.n_states, env.n_actions))
    pi_s = Q - np.max(Q, axis=1)[:, None]
    #pi_s[np.where(pi_s == 0)] = 1
    pi_s[np.where((-tol <= pi_s) & (pi_s <= tol))] = 1
    pi_s[np.where(pi_s <= 0)] = 0
    pi_s = pi_s/pi_s.sum(axis=1)[:, None]
    #print("pi_s=", pi_s)
    #print("pi_s shape=", pi_s.shape, "pi_s type=", type(pi_s), "pi_s[0, 0] type=", type(pi_s[0, 0]))

    return Q, V, pi_d, pi_s
#enddef


def span_norm(v):
    return np.max(v) - np.min(v)
#enddef
def averaged_valueIteration(env, reward, tol=1e-6):
    V = np.zeros((env.n_states))
    Q = np.zeros((env.n_states, env.n_actions))
    iter = 0
    while True:
        iter +=1
        V_old = copy.deepcopy(V)
        #print("before for loop")
        for a in range(env.n_actions):
            Q[:, a] = reward[:, a] + env.T_sparse_list[a].dot(V)
        #print("after for loop")
        V = np.max(Q, axis=1)
        #print("iter=", iter, " tol=", abs(np.linalg.norm(V - V_old, np.inf)))
        if span_norm(V - V_old) < tol:
            break
    #endwhile
    # For a deterministic policy
    pi_d = np.argmax(Q, axis=1)
    pi_s = Q - np.max(Q, axis=1)[:, None]
    #pi_s[np.where(pi_s == 0)] = 1
    pi_s[np.where((-tol <= pi_s) & (pi_s <= tol))] = 1
    pi_s[np.where(pi_s <= 0)] = 0
    pi_s = pi_s/pi_s.sum(axis=1)[:, None]
    return V, pi_d, pi_s
#enddef
def extended_averaged_value_iteration(env, r_hat, conf_r, p_hat, conf_p, tol=1e-5):
    n_states = env.n_states
    n_actions = env.n_actions
    start = time.time()
    print("extended value iteration started")
    u = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    r_tilda = r_hat + conf_r

    while True:
        u_old = copy.deepcopy(u)
        for s in range(n_states):
            for a in range(n_actions):
                p_hat_vec = p_hat[s, :, a]
                conf_p_scalar = conf_p[s, a]

                Q[s, a] = r_tilda[s, a] + compute_inner_maximum(n_states, u_old, p_hat_vec, conf_p_scalar).dot(u_old)
        u = np.max(Q, axis=1)
        # print(span_norm(u-u_old))
        if span_norm(u - u_old) < tol:
            break
    pi_d = np.argmax(Q, axis=1)
    end = time.time()
    print("extendd valur iteration ended:")
    print("time =", end - start)
    return pi_d
#enddef



def compute_inner_maximum(n_states, u, p_hat_vector, conf_p_scalar):
    # print("inside compute_inner_maximum")
    S = numpy.argsort(u)[::-1]  # sort descending
    p = np.zeros(n_states)
    for i in range(len(S)):
        #1)
        if i == 0:
            p[S[i]] = min(1.0, p_hat_vector[S[i]] + conf_p_scalar/2)
        else:
            p[S[i]] = p_hat_vector[S[i]]
    #2)
    l = n_states-1
    #3)
    while(np.sum(p) > 1 + 1e-10):
        #(a)
        p[S[l]] = max(0, 1 - (sum(p) - p[S[l]]))
        #(b)
        l = l - 1
    return p
#enddef

# def compute_averaged_reward_given_policy(env, policy):
#
#     if len(policy.shape) == 1:
#         policy = convert_det_to_stochastic_policy(env, policy)
#
#     stationary_dist = get_stationary_dist(env, policy)
#     P_pi_sparse,_ = get_P_pi_sparse(env, policy)
#     expected_reward = 0
#     for s in range(env.n_states):
#         for a in range(env.n_actions):
#             for n_s in range(env.n_states):
#                 expected_reward += stationary_dist[s] * env.T[s, n_s, a] * env.reward[s, a]
#     return expected_reward
# #enddef

def compute_averaged_reward_given_policy(env, reward, policy):
    #
    # if len(policy.shape) == 1:
    #     policy = convert_det_to_stochastic_policy(env, policy)
    stationary_dist = get_stationary_dist(env, policy)
    expected_reward = 0
    for s in range(env.n_states):
        expected_reward += stationary_dist[s] * reward[s, policy[s]]
    return expected_reward
#enddef
def convert_det_to_stochastic_policy(env, deterministicPolicy):
    # Given a deterministic Policy, I will return a stochastic policy
    stochasticPolicy = np.zeros((env.n_states, env.n_actions))
    if env.terminal_state == 1:
        n_states = env.n_states-1
    else:
        n_states = env.n_states
    for i in range(n_states):
        stochasticPolicy[i][deterministicPolicy[i]] = 1
    return stochasticPolicy
#enddef

def get_P_pi(env, policy):
    if policy.ndim == 1:
        policy = convert_det_to_stochastic_policy(env, policy)
    P_pi = np.zeros((env.n_states, env.n_states))
    for n_s in range(env.n_states):
        for a in range(env.n_actions):
            P_pi[:, n_s] += policy[:, a] * env.T[:, n_s, a]

    return P_pi
# enddef

def computeValueFunction_given_policy_linear_program(env, pi, reward):
    states_count = env.n_states
    actions_count = env.n_actions
    gamma = env.gamma
    R = reward
    P = env.T
    T = np.zeros((states_count, states_count))
    for x in range(states_count):
        for y in range(states_count):
            T[x, y] = P[x, y, pi[x]]

    R_pi = np.zeros(states_count)
    for x in range(states_count):
        R_pi[x] = R[x, [pi[x]]]


    if gamma == 1:
        A = np.zeros((states_count, states_count))
        A[0:states_count - 1, :] = (np.identity(states_count) - T)[1:, :]
        A[states_count - 1, 0] = 1

        b = np.zeros(states_count)
        b[0:states_count - 1] = (R_pi * np.ones(states_count))[1:]
    else:
        A = np.identity(states_count) - gamma * T
        b = R_pi * np.ones(states_count)
    V_pi = np.linalg.solve(A, b)
    return V_pi

def computeValueFunction_given_policy(env, reward, expected_reward, policy, tol=1e-5):
    # Given a policy (could be either deterministic or stochastic), I return the Value Function
    # Using the Bellman Equations
    # Let's check if this policy is deterministic or stochastic
    if len(policy.shape) == 1:
        changed_policy = convert_det_to_stochastic_policy(env, policy)
    else:
        changed_policy = policy

    P_pi = get_P_pi(env, changed_policy)

    # Converting this T to a sparse matrix
    P_pi_sparse = sparse.csr_matrix(P_pi)
    # Some more initialisations
    V = np.zeros(env.n_states)
    Q = np.zeros((env.n_states, env.n_actions))
    #reward = env.get_reward_for_given_w(w)
    iter = 0
    # Bellman Equation
    while True:
        iter += 1
        V_old = copy.deepcopy(V)
        for a in range(env.n_actions):
            Q[:, a] = reward[:, a] - expected_reward + P_pi_sparse.dot(V)
        V = np.max(Q, axis=1)
        if span_norm(V - V_old) < tol:
            break
    # Converged. let's return the V
    return V
#enddef

def computeValueFunction_bellmann(env, policy, reward, tol=1e-6):
    # Given a policy (could be either deterministic or stochastic), I return the Value Function
    # Using the Bellman Equations
    # Let's check if this policy is deterministic or stochastic
    if len(policy.shape) == 1:
        changed_policy = convert_det_to_stochastic_policy(env, policy)
    else:
        changed_policy = policy

    T_pi = get_T_pi(env, changed_policy)

    # Converting this T to a sparse matrix
    T_pi_sparse = sparse.csr_matrix(T_pi)
    # Some more initialisations
    V = np.zeros((env.n_states))
    Q = np.zeros((env.n_states, env.n_actions))
    #reward = env.get_reward_for_given_w(w)
    iter = 0
    # Bellman Equation
    while True:
        iter += 1
        V_old = copy.deepcopy(V)
        for a in range(env.n_actions):
            Q[:, a] = reward[:, a] + env.gamma * T_pi_sparse.dot(V)
        V = np.max(Q, axis=1)
        if abs(np.linalg.norm(V - V_old, np.inf)) < tol:
            break
    # Converged. let's return the V
    return V

def compute_Q_V_Function_given_policy(env, policy, reward, tol=1e-6):
    V_pi = computeValueFunction_given_policy_linear_program(env, policy, reward)
    Q_pi = np.zeros((env.n_states, env.n_actions))

    for s in range(env.n_states):
        for a in range(env.n_actions):
            sum_over_s_n = env.gamma*env.T[s, :, a].dot(V_pi[:])
            Q_pi[s, a] = reward[s, a] + sum_over_s_n

    return Q_pi, V_pi

def computeFeatureSVF_bellmann(env, policy, tol=1e-6):
    # Given a policy, Return State Visitation Frequencies and feature Expectation
    # Using Bellman Equation
    # Let's ensure we have a stochastic policy, if not lets convert
    if len(policy.shape) == 1:
        changed_policy = convert_det_to_stochastic_policy(env, policy)
    else:
        changed_policy = policy
    # Creating a T matrix for the policy
    T_pi = get_T_pi(env, changed_policy)

    if env.terminal_state == 1:
        T_pi[-1, :] = 0

    # Converting T to a sparse matrix
    #start = time.time()
    T_pi_sparse = sparse.csr_matrix(T_pi.transpose())
    #print("time to create sparse matrix in svf_bellmann=", time.time() - start)
    # Some initialisations
    SV = np.zeros((env.n_states))
    init_d = np.ones(env.n_states)/env.n_states
    iter = 0
    # Bellman Equation
    while True:
        iter += 1
        SV_old = copy.deepcopy(SV)
        SV[:] = init_d + env.gamma * T_pi_sparse.dot(SV[:])
        if abs(np.linalg.norm(SV - SV_old)) < tol:
            break
    return SV





def compute_B_pi(env, reward, expected_reward, policy):
    B_pi = np.zeros((env.n_states))

    V_pi = computeValueFunction_given_policy(env, reward, expected_reward, policy)

    P_pi = get_P_pi(env, policy)

    # Converting this T to a sparse matrix
    P_pi_sparse = sparse.csr_matrix(P_pi)
    # while True:
    #     B_pi_old = copy.deepcopy(B_pi)
    B_pi[:] = P_pi_sparse.dot(V_pi)
    return B_pi
#enddef

def get_T_pi(env, policy):
    T_pi = np.zeros((env.n_states, env.n_states))
    for n_s in range(env.n_states):
        for a in range(env.n_actions):
            T_pi[:, n_s] += policy[:, a] * env.T[:, n_s, a]

    return T_pi
# enddef

def calc_reachtimes(env, pi):
    P = env.T
    gamma = env.gamma
    states_count = env.n_states
    T = np.zeros((states_count, states_count))
    for x in range(states_count):
        for y in range(states_count):
            T[x, y] = P[x, y, pi[x]]

    reach_times = np.zeros((states_count, states_count))
    for s2 in range(states_count):
        A = np.delete(T, s2, 1)
        A = np.delete(A, s2, 0)
        h = np.linalg.inv(np.identity(states_count - 1) - gamma * A) @ np.ones(states_count - 1)
        h = np.insert(h, s2, 0)
        reach_times[:, s2] = h

    return reach_times

def calc_reachtimes_with_T_pi(env, pi):

    P_pi = get_P_pi(env, pi)
    gamma = env.gamma
    states_count = env.n_states
    T = np.zeros((states_count, states_count))
    for x in range(states_count):
        for y in range(states_count):
            T[x, y] = P_pi[x, y]

    reach_times = np.zeros((states_count, states_count))
    for s2 in range(states_count):
        A = np.delete(T, s2, 1)
        A = np.delete(A, s2, 0)
        h = np.linalg.inv(np.identity(states_count - 1) - gamma * A) @ np.ones(states_count - 1)
        h = np.insert(h, s2, 0)
        reach_times[:, s2] = h

    return reach_times




########################################
if __name__ == "__main__":
    pass






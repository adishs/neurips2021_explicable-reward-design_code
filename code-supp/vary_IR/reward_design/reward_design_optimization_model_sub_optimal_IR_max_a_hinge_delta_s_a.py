import numpy as np
import copy
import sys
import cvxpy as cp
import itertools as it



def reward_design_model_based(env_0,  pi_d, pi_s, R_max, H_set, s_active =None,
                    delta_s_a_array=None, C_1=1, C_2=0, dict_s_opt_actions_arr=None,
                              state_only_reward_flag=False, is_delta_s_const=False,
                              upper_c_for_delt_s=5, tol=1e-10):

    if s_active is None:
        s_active = env_0.goal_state# last or second last state as goal

    n_states = env_0.n_states if env_0.terminal_state == 0 else env_0.n_states - 1 #chech if we have terminal state
    n_states_all = env_0.n_states

    n_actions = env_0.n_actions
    gamma = env_0.gamma
    #variable declaration
    # reward \in R^|S|*|A|
    R = cp.Variable((n_states_all * n_actions))

    if H_set is None:
        H_set = []

    set_of_pi_d = get_set_of_determinisric_policies(env_0, pi_d, dict_s_opt_actions_arr)

    ##### delta_s_array =====================
    delta_s_a_array = np.array(delta_s_a_array)
    delta_s_array = get_delta_s_given_delta_s_a(delta_s_a_array, pi_d, pi_s)
    delta_s_array = np.array(delta_s_array) / (1 + 1e-5)
    if is_delta_s_const:
        if upper_c_for_delt_s is not None:
            delta_s_array = np.minimum(delta_s_array, upper_c_for_delt_s)
        else:
            print("upper_c_for_delt_s is NONE and is_delta_s_const Flag is True")
            exit(0)
    constraints = []
    delta_s_eps_h_s_array_diff = []


    for i, opt_pi_d_t in enumerate(set_of_pi_d):
        I_pi_star = calculate_I_pi_star(env_0, opt_pi_d_t)
        P_pi_star = calculate_P_pi_star(env_0, opt_pi_d_t)
        I = np.eye(n_states_all * n_actions)


        #global optimality
        A = (I_pi_star - I) @ np.linalg.inv((I - gamma * P_pi_star))
        b = np.zeros(n_states_all * n_actions)
        # set zero for optimal actions
        for s in range(n_states_all):
            for a in range(n_actions):
                if pi_s[s, a] == 0:
                    b[s * n_actions + a] = delta_s_array[s]

        A_x = A @ R

        # global optimality constraints
        for i in range(A_x.shape[0]):
            cons = (A_x[i] >= b[i])
            constraints.append(cons)

        # Q_H constraints for every H
        for i, H in enumerate(H_set):
            accumulator = get_A_local_h(env_0, P_pi_star, I, H=H)

            A_local = (I_pi_star - I) @ (accumulator)

            A_x_local = A_local @ R

            # calculate eps_h array
            for s in range(n_states_all):
                s_a_array = []
                for a in range(n_actions):
                    if pi_s[s, a] == 0:
                        s_a_array.append(delta_s_a_array[s, a] - A_x_local[s * n_actions + a])
                if len(s_a_array) != 0:
                    delta_s_eps_h_s_array_diff.append(cp.max(cp.hstack(s_a_array)))


    # converte back to cvxpy variable
    delta_s_eps_h_s_array_diff = cp.hstack(delta_s_eps_h_s_array_diff)

    #  R_max bounds
    for s in range(n_states_all):
        for a in range(n_actions):
            cons_1 = (R[s * n_actions + a] >= -R_max)
            cons_2 = (R[s * n_actions + a] <= R_max)
            constraints.append(cons_1)
            constraints.append(cons_2)

    # sparsity constraints
    for s in range(n_states_all):
        for a in range(n_actions):
            if s not in s_active:
                cons = (R[s * n_actions + a] == 0)
                constraints.append(cons)

    if state_only_reward_flag:
        for s in range(n_states_all):
            for a in range(1, n_actions):
                cons = (R[s * n_actions + a] == R[s * n_actions + 0])
                constraints.append(cons)

    IR = - (1 / len(set_of_pi_d))* (1 / len(H_set)) * (1 / n_states_all) * \
           cp.sum(cp.pos(delta_s_eps_h_s_array_diff))


    obj = cp.Minimize(-C_1 * IR + C_2 * cp.norm(R, 1))
    prob = cp.Problem(obj, constraints)
    # Solve the problem
    prob.solve(solver=cp.ECOS, max_iters=1000, feastol=tol, abstol=tol, reltol=tol)
    # prob.solve()
    # prob.solve()
    obj_value = copy.deepcopy(prob.value)
    # get solution
    R_sol = R.value

    # round solution to the precision
    R_sol = np.round(R_sol, 8)

    ## zero out the elemets close to zero
    R_sol[np.where((-tol <= R_sol) & (R_sol <= tol))] = 0.0
    # get final solutions
    reward = R_sol.reshape(n_states_all, n_actions)
    # epsilon_inf = R_sol[n_states * n_actions]
    # epsilon_H_arr = R_sol[n_states * n_actions + 1:]
    return obj_value, reward
# enddef


def get_A_local_h(env_0, P_pi_star, I, H):
    #local (H horizon) optimality constraints
    accumulator = copy.deepcopy(I)
    accumulator_P_star_mult = copy.deepcopy(I)
    gamma = env_0.gamma

    for h in range(1, H+1):
        accumulator_P_star_mult = accumulator_P_star_mult @ P_pi_star
        accumulator += (gamma**h) * accumulator_P_star_mult

    return accumulator
#enddef

def get_set_of_determinisric_policies(env_0, pi_d, dict_s_opt_actions):
    n_states = env_0.n_states
    opt_action_set = []

    for s in range(n_states):
        if s in dict_s_opt_actions:
            opt_action_set.append(dict_s_opt_actions[s])
        else:
            opt_action_set.append([pi_d[s]])
    set_det_policies_tuple = list(it.product(*opt_action_set))

    # convert to array
    set_det_policies = []
    for pi_d in set_det_policies_tuple:
        set_det_policies.append(np.array(pi_d))
    return set_det_policies


# enddef

def get_mat_for_concatenation(env_0, pi_t, j, H_set, delta_s_array):
    n_states = env_0.n_states
    n_actions = env_0.n_actions
    mat_for_concat_H = np.zeros((n_states*n_actions, n_states*n_actions))
    #set zero for optimal actions
    for s in range(n_states):
        for a in range(n_actions):
            if a != pi_t[s]:
                mat_for_concat_H[s * n_actions + a, s * n_actions + a] = - delta_s_array[s] / 2
    return mat_for_concat_H
#enddef

def calculate_I_pi_star(env, pi_star):
    n_states = env.n_states
    n_actions = env.n_actions
    I_pi_star = np.zeros((n_states * n_actions, n_states * n_actions))

    for s in range(n_states):
        opt_act = pi_star[s]
        I_pi_star[s * n_actions:s * n_actions + n_actions, s * n_actions + opt_act] = 1
        # print(s*n_actions,":",s*n_actions+n_actions)
        # print(s*n_actions + opt_act)
        # input()
    return I_pi_star
# enddef

def calculate_P_pi_star(env, pi_star):
    n_states = env.n_states
    n_actions = env.n_actions
    P_0 = env.T

    P_pi_star = np.zeros((n_states * n_actions, n_states * n_actions))

    for s in range(n_states):
        for a in range(n_actions):
            for s_n in range(n_states):
                for a_n in range(n_actions):
                    P_pi_star[s * n_actions + a, s_n * n_actions + a_n] = \
                        P_0[s, s_n, a] if a_n == pi_star[s_n] else 0
    return P_pi_star
# enddef

def get_delta_s_given_delta_s_a(delta_s_a, pi_target_d, pi_target_s):
    delta_s_a = np.array(delta_s_a)
    delta_s_array = []
    for s in range(delta_s_a.shape[0]):
        s_a_array = []
        for a in range(delta_s_a.shape[1]):
            if pi_target_s[s, a] == 0:
                s_a_array.append(np.min(delta_s_a[s, a]))
        if len(s_a_array) == 0:
            s_a_array.append(0)
        delta_s_array.append(min(s_a_array))
    return delta_s_array

def get_important_states_set_policies_model_based(M_0, pi_d, pi_s, R_max, H_set, s_active=None,
                                      delta_s_array=None, C_1=1.0, C_2=1.0, n_nonzero_states=1, candidate_states=None,
                                      dict_s_opt_actions_arr=None, state_only_reward_flag=False, tol=1e-10):
    if s_active is None:
        s_active = [M_0[0] - 1 - M_0[5]]  # last or second last state as goal

    s_active_set_chosen = copy.deepcopy(s_active)
    for i in range(1, n_nonzero_states):
        min_value = np.inf
        state_chosen = None
        for s in candidate_states:
            if s not in s_active_set_chosen:
                s_active_candidate = np.concatenate((s_active_set_chosen, [s]))
                print("===================")
                print("S_active_chosen: ", s_active_set_chosen)
                print("S_candidate: ", s)
                obj_value, _ = reward_design_model_based(M_0, pi_d=pi_d, pi_s=pi_s, R_max=R_max, H_set=H_set,
                                                          s_active=s_active_candidate,
                                                          delta_s_array=delta_s_array, C_1=C_1, C_2=C_2,
                                                          dict_s_opt_actions_arr=dict_s_opt_actions_arr,
                                                         state_only_reward_flag=state_only_reward_flag, tol=tol)
                print("obj_value: ", obj_value)
                print("===================")

                if min_value > obj_value:
                    min_value = copy.deepcopy(obj_value)
                    state_chosen = copy.deepcopy(s)
        s_active_set_chosen.append(state_chosen)

    obj_value, R_sol = reward_design_model_based(M_0, pi_d=pi_d, pi_s=pi_s, R_max=R_max, H_set=H_set,
                                                  s_active=s_active_set_chosen,
                                                  delta_s_array=delta_s_array, C_1=C_1, C_2=C_2,
                                                  dict_s_opt_actions_arr=dict_s_opt_actions_arr,
                                                 state_only_reward_flag=state_only_reward_flag, tol=tol)

    return s_active_set_chosen, R_sol


# enddef




#enddef


########################################
if __name__ == "__main__":
    pass

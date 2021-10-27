import numpy as np
import copy
from scipy import sparse
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.tri import Triangulation
import math
import MDPSolver
import sys
import os



class Environment:
    def __init__(self, env_args):

        # initialise MDP parameters
        self.actions = {"up": 0, "left": 1, "down": 2, "right": 3}
        self.actions_names = ["up", "left", "down", "right"]
        self.n_actions = len(self.actions)
        self.gridsizefull = env_args["gridsizefull"]
        self.R_max = env_args["R_max"]
        self.gamma = env_args["gamma"]
        self.terminal_state = env_args["terminalState"]
        self.randomMoveProb = env_args["randomMoveProb"]
        self.wall_horizontal, self.wall_vertical, self.gates = self.compute_walls()
        self.gate_states = self.get_gate_states(self.gates)
        self.n_states = self.gridsizefull * self.gridsizefull if self.terminal_state == 0 \
            else self.gridsizefull * self.gridsizefull + 1
        self.InitD = self.get_InitD()
        self.reward = self.get_reward()
        self.T = self.get_transition_matrix(self.randomMoveProb)
        self.T_sparse_list = self.get_transition_sparse_list()
        self.goal_state = self.get_goal_states()
        self.graph_dict = self.get_graph()
        # self.distance_array_BFS = self.generate_distance_from_graph_using_BFS()
        self.distance_array_reachtimes = self.generate_distance_from_reachtimes()

    #enddef

    def get_InitD(self):
        InitD = np.zeros(self.n_states)
        init_state = self.get_state_from_coord(1, 1)
        InitD[init_state] = 1
        return InitD
    #enddef

    def get_reward(self):
        reward = np.zeros((self.n_states, self.n_actions))
        if self.terminal_state == 0:
            reward[-1, 3] = self.R_max
        else:
            reward[-2, 3] = self.R_max
        return reward
    #enndef

    def get_goal_states(self):

        reward_soom_over_states = np.sum(self.reward, axis=1)
        goal_state = np.nonzero(reward_soom_over_states)[0]
        return goal_state
    #enddef

    def compute_wall_states(self, wall):
        wall_states = []
        for x, y in wall:
            wall_states.append(self.get_state_from_coord(x, y))
        return list(set(wall_states))
    #enddef

    def get_gate_states(self, gates):
        gate_states = []
        if self.gridsizefull%2==1:
            for x, y in gates:
                gate_states.append(self.get_state_from_coord(x, y))
            gate_states = sorted(gate_states)
            gate_states[0] -= 1
            gate_states[1] = gate_states[1]
            gate_states[2] -= self.gridsizefull
            gate_states[3] -= 1
        else:
            for x, y in gates:
                gate_states.append(self.get_state_from_coord(x, y))
            gate_states = sorted(gate_states)
            gate_states[0] = gate_states[0]
            gate_states[1] += self.gridsizefull
            gate_states[2] = gate_states[2]
            gate_states[3] = gate_states[3]

        return gate_states
    #enddef

    def get_next_state(self, s_t, a_t):
        next_state = np.random.choice(np.arange(0, self.n_states, dtype="int"), size=1, p=self.T[s_t, :, a_t])[0]
        return next_state
    #enddef

    def compute_walls(self):
        '''
        Returns:
            (list): Contains (x,y) pairs that define wall locations.
        '''
        walls_horizontal = []
        walls_vertical = []
        gates = []

        half_width = math.ceil(self.gridsizefull / 2.0)
        half_height = math.ceil(self.gridsizefull / 2.0)

        half_height_const = self.gridsizefull // 2 + 1
        half_width_const = self.gridsizefull//2

        # Wall from left to middle.
        for i in range(1, self.gridsizefull + 1):
            if i == half_width:
                half_height -= 1
            if i + 1 == math.ceil(self.gridsizefull / 3.0) or\
                    i == math.ceil(2 * (self.gridsizefull + 2) / 3.0):
                gates.append((i-1, half_height-1))
                continue

            walls_horizontal.append((i-1, half_height_const-1))

        # Wall from bottom to top.
        # half_width_const = copy.deepcopy(half_width)
        for j in range(1, self.gridsizefull + 1):
            if j + 1 == math.ceil(self.gridsizefull / 3.0) or \
                    j == math.ceil(2 * (self.gridsizefull + 2) / 3.0):
                gates.append((half_width-1, j-1))
                continue
            walls_vertical.append((half_width_const, j-1))


        return walls_horizontal, walls_vertical, gates
    #enddef


    def get_transition_matrix(self, randomMoveProb=0.3):
        states = self.n_states
        P = np.zeros((states, states, self.n_actions))
        for s in range(self.gridsizefull*self.gridsizefull):
            possible_actions = self.get_possible_actions_within_grid(s)
            next_states = self.get_next_states(s, possible_actions)
            for a in range(self.n_actions):
                if a in possible_actions:
                    n_s = int(next_states[np.where(possible_actions == a)][0]) #direct next state
                    P[s, n_s, a] = 1.0 - randomMoveProb #set transition probability
                    next_states_copy = np.setdiff1d(next_states, n_s) # alll other reachable states except direct next states
                    if len(next_states_copy) == 0: #if there is no other reachable next state
                        next_states_copy = np.array([s]) #set state itself
                    for i in next_states_copy:
                        P[s, i, a] += randomMoveProb / len(next_states_copy) #set uniform randomMove
                elif a not in possible_actions: # not defined action
                    for i in next_states:
                        # i_next_wall_state = self.state_to_wall_state(i)
                        P[s, i, a] += randomMoveProb / len(next_states)
                    P[s, s, a] += 1 - randomMoveProb

        if self.terminal_state == 1:
            #0th state
            P[0, :, :] = 0
            #UP action
            P[0, self.gridsizefull, 0] = 1-randomMoveProb
            P[0, 1, 0] = randomMoveProb/2
            P[0, -1, 0] = randomMoveProb/2

            #LEFT action
            P[0, -1, 1] = 1-randomMoveProb
            P[0, 1, 1] = randomMoveProb/2
            P[0, self.gridsizefull, 1] = randomMoveProb/2
            #DOWN action
            P[0, -1, 2] = 1-randomMoveProb
            P[0, 1, 2] = randomMoveProb/2
            P[0, self.gridsizefull, 2] = randomMoveProb/2

            #Right action
            P[0, 1, 3] = 1-randomMoveProb
            P[0, -1, 3] = randomMoveProb/2
            P[0, self.gridsizefull, 3] = randomMoveProb/2
            ################
            #goal state
            P[-2, :, :] = 0
            #up action from the state brings you to terminal
            P[-2:, -1, 0] = 1-randomMoveProb
            P[-2:, -3, 0] = randomMoveProb/2
            P[-2:, -self.gridsizefull-1, 0] = randomMoveProb/2

            P[-2:, -3, 1] = 1-randomMoveProb
            P[-2:, -1, 1] = randomMoveProb/2
            P[-2:, -self.gridsizefull-1, 1] = randomMoveProb/2

            P[-2:, -self.gridsizefull-1, 2] = 1-randomMoveProb
            P[-2:, -1, 2] = randomMoveProb/2
            P[-2:, -3, 2] = randomMoveProb/2

            #right action from the state brings you to terminal
            P[-2:, -1, 3] = 1-randomMoveProb
            P[-2:, -3, 3] = randomMoveProb/2
            P[-2:, -self.gridsizefull-1, 3] = randomMoveProb/2

            #terminal self transition
            P[-1, :, :] = 0
            P[-1, -1, :] = 1

        return P
    #enddef

    def get_transition_sparse_list(self):
        T_sparse_list = []
        for a in range(self.n_actions):
            T_sparse_list.append(sparse.csr_matrix(self.T[:, :, a]))
        return T_sparse_list
    # endef

    def state_to_wall_state(self, state):
        state_wall = np.array([a[0] + a[1] * self.gridsizefull for a in self.wall])
        # print(state_wall)
        n_less = len(np.where(state_wall <= state)[0])
        return state - n_less


    def get_possible_actions_within_grid(self, state):
        # Given a state, what are the possible actions from it
        possible_actions = []
        state_x, state_y = state % self.gridsizefull, state // self.gridsizefull
        # print(state_x, state_y)
        # print((state_x-1, state_y) in self.wall)
        if ((state_y < self.gridsizefull-1) and (state_x, state_y+1) not in self.wall_horizontal): #UP action is not allowed from down state of the horizontal wall
            possible_actions.append(self.actions["up"])
        if ((state_y > 0) and (state_x, state_y) not in self.wall_horizontal): #DOWN action is not allowed from horizontal wall state
            possible_actions.append(self.actions["down"])
        if ((state_x > 0) and (state_x, state_y) not in self.wall_vertical): #LEFT action is not allowed from the vertical wall_state
            possible_actions.append(self.actions["left"])
        if ((state_x < self.gridsizefull - 1) and (state_x+1, state_y) not in self.wall_vertical): #RIGHT action is not allwed from the left side of the vertical wall
            possible_actions.append(self.actions["right"])
        # possible_actions.append(self.actions["stay"])
        possible_actions = np.array(possible_actions, dtype=np.int)
        return possible_actions
    #enddef

    def get_next_states(self, state, possible_actions):
        # Given a state, what are the posible next states I can reach
        next_state = []
        state_x, state_y = state % self.gridsizefull, state // self.gridsizefull
        for a in possible_actions:
            if a == 0: next_state.append((state_y+1) * self.gridsizefull + state_x)
            elif a == 1: next_state.append(state_y * self.gridsizefull + state_x-1)
            elif a == 2: next_state.append((state_y-1) * self.gridsizefull + state_x)
            elif a == 3: next_state.append(state_y * self.gridsizefull + state_x + 1)
            # else: next_state.append(state)
        next_state = np.array(next_state, dtype=np.int)
        return next_state

    def is_wall(self, x, y):
        state_coord = (x, y)
        if state_coord in self.wall_horizontal or state_coord in self.wall_vertical:
            return True
        return False
    #enddef

    def is_wall(self, s):
        state_x, state_y = s % self.gridsizefull, s // self.gridsizefull
        state_coord = (state_x, state_y)
        if state_coord in self.wall_horizontal or state_coord in self.wall_vertical:
            return True
        return False
    #enddef

    def get_state_from_coord(self, x, y):
        # state_wall = np.array([a[0] + a[1]*self.gridsizefull for a in self.wall])
        # print(state_wall)
        orig_state = y * self.gridsizefull + x
        # n_less = len(np.where(state_wall<=orig_state)[0])
        return orig_state
    #enddef

    def convert_det_to_stochastic_policy(self, deterministicPolicy):
        # Given a deterministic Policy, I will return a stochastic policy
        n_states = self.gridsizefull*self.gridsizefull
        stochasticPolicy = np.zeros((n_states, self.n_actions))
        for i in range(n_states):
            stochasticPolicy[i][deterministicPolicy[i]] = 1
        return stochasticPolicy

    # enddef

    def draw(self, V, pi, reward, show, strname, fignum):
        f = fignum
        n_states = self.n_states
        # plt.figure(f)

        # pi = copy.deepcopy(pi)
        # pi = pi.reshape(self.gridsizefull, self.gridsizefull)
        # pi = pi.flatten()


        if len(pi.shape) == 1:
            pi = self.convert_det_to_stochastic_policy(pi)
        # plt.pcolor(reward)
        # plt.title(strname + "Reward")
        # plt.colorbar()

        # print(pi.shape)

        if self.terminal_state == 1:
            V = copy.deepcopy(V[:-1])
            pi = np.delete(pi, (-1), axis=0)
            n_states = self.n_states-1

        f += 1
        if V is not None:
            plt.figure(f)
            reshaped_Value = copy.deepcopy(V.reshape((self.gridsizefull, self.gridsizefull)))
            plt.pcolor(reshaped_Value)
            plt.colorbar()
            x = np.linspace(0, self.gridsizefull - 1, self.gridsizefull) + 0.5
            y = np.linspace(0, self.gridsizefull - 1, self.gridsizefull) + 0.5
            X, Y = np.meshgrid(x, y)
            zeros = np.zeros((self.gridsizefull, self.gridsizefull))
            if pi is not None:
                for a in range(self.n_actions):
                    pi_ = np.zeros(n_states)
                    for s in range(n_states):
                        pi_[s] = 0.45*pi[s, a]/np.max(pi[s, :])

                    pi_ = (pi_.reshape(self.gridsizefull, self.gridsizefull))
                    if a == 0:
                        plt.quiver(X, Y, zeros, pi_, scale=1, units='xy')
                    elif a == 1:
                        plt.quiver(X, Y, -pi_, zeros, scale=1, units='xy')
                    elif a == 2:
                        plt.quiver(X, Y, zeros, -pi_, scale=1, units='xy')
                    elif a == 3:
                        plt.quiver(X, Y, pi_, zeros, scale=1, units='xy')
            plt.title(strname + "Opt values and policy")
        if(show == True):
            #print(" show is true")
            plt.show()
    #enddef


    def plot_reward(self, reward, fignum =1, show=True, str_name = "name"):
        M, N = self.gridsizefull, self.gridsizefull  # e.g.  columns,  rows
        wall_state_value = np.min(reward) - 1

        if self.terminal_state == 1:
            reward = np.delete(reward, (-1), axis=0)


        # create some demo data for North, East, South, West
        # valuesN = np.array([[0] * self.gridsizefull, [.7] * self.gridsizefull, [0.5] * self.gridsizefull, [1] * self.gridsizefull])
        valuesN = reward[:, 0].reshape(M, N)
        valuesE = reward[:, 1].reshape(M, N)
        valuesS = reward[:, 2].reshape(M, N)
        valuesW = reward[:, 3].reshape(M, N)
        values = [valuesN, valuesE, valuesS, valuesW]
        # print(valuesN)
        # print(valuesE)
        # exit(0)

        xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
        xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
        x = np.concatenate([xv.ravel(), xc.ravel()])
        y = np.concatenate([yv.ravel(), yc.ravel()])
        cstart = (M + 1) * (N + 1)  # indices of the centers

        trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M) for j in range(N) for i in range(M)]
        trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M) for j in range(N) for i in
                      range(M)]
        trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M) for j in range(N) for i in
                      range(M)]
        trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M) for j in range(N) for i in range(M)]
        triangul = [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]

        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
        fig.num = fignum
        # fig.subplots_adjust(left=0.4, right=0.5, bottom=0.4, top=0.5)

        imgs = [ax.tripcolor(t, val.ravel(), cmap='RdYlGn', vmin=np.min(reward), vmax=np.max(reward), edgecolors="black")
                for t, val in zip(triangul, values)]
        for val, dir in zip(values, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
            for i in range(M):
                for j in range(N):
                    v = val[j, i]
                    ax.text(i + 0.3 * dir[1], j + 0.3 * dir[0], f'{v:.2f}' if v != wall_state_value else "-", color='black' if v != wall_state_value else 'w',
                            ha='center', va='center')
        cbar = fig.colorbar(imgs[0], ax=ax)
        plt.title(str_name)
        if show:
            plt.show()
    #enddef


    def get_graph(self):
        graph_dict = {}
        for s in range(self.n_states):
            transition_s_n_array = []
            for a in range(self.n_actions):
                transition_s_n_array = np.concatenate((transition_s_n_array,
                                                       np.setdiff1d(np.nonzero(self.T[s, :, a]), transition_s_n_array)))
            transition_s_n_array = np.array(np.setdiff1d(transition_s_n_array, [s]), dtype=int)
            graph_dict["{}".format(s)] = [str(i) for i in transition_s_n_array]
        return graph_dict
    #enddef

    def BFS_SP(self, start, goal):
        graph = self.graph_dict
        explored = []

        # Queue for traversing the
        # graph in the BFS
        queue = [[start]]

        # If the desired node is
        # reached
        if start == goal:
            return [start]

        # Loop to traverse the graph
        # with the help of the queue
        while queue:
            path = queue.pop(0)
            node = path[-1]

            # Condition to check if the
            # current node is not visited
            if node not in explored:
                neighbours = graph[node]

                # Loop to iterate over the
                # neighbours of the node
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)

                    # Condition to check if the
                    # neighbour node is the goal
                    if neighbour == goal:
                        return new_path
                explored.append(node)

        # Condition when the nodes
        # are not connected
        # print("So sorry, but a connecting" "path doesn't exist :(")
        return None
    #enddef

    def generate_distance_from_graph_using_BFS(self):

        distance_array = np.ones((self.n_states, self.n_states)) * sys.maxsize

        for s in range(self.n_states):
            for s_n in range(self.n_states):
                path_state_array = self.BFS_SP(str(s), str(s_n))

                if path_state_array is not None:
                    distance_array[s, s_n] = (len(path_state_array) - 1)

        if self.terminal_state == 1:
            distance_array[-1] = np.zeros(self.n_states)
            distance_array[-2] = np.zeros(self.n_states)

        return distance_array
    #enddef

    def distance_fun(self, s_1, s_2):
        normalization = (self.n_states * np.max(self.distance_array_reachtimes))
        return self.distance_array_reachtimes[s_1, s_2] / normalization
    #enddef

    def generate_distance_from_reachtimes(self):
        pi_s_random = np.ones((self.n_states, self.n_actions)) / self.n_actions #random policy
        T_reach_times = MDPSolver.calc_reachtimes_with_T_pi(self, pi_s_random)
        return T_reach_times
    #enddef


#endclass



def write_into_file(accumulator, exp_iter, out_file_name="output"):
    directory = 'results/{}'.format(out_file_name)
    filename = "convergence" + '_' + str(exp_iter) + '.txt'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filepath = directory + '/' + filename
    print("output file name  ", filepath)
    f = open(filepath, 'w')
    for key in accumulator:
        f.write(key + '\t')
        temp = list(map(str, accumulator[key]))
        for j in temp:
            f.write(j + '\t')
        f.write('\n')
    f.close()
#enddef

def facility_function(env, S, Z):
    sum_over_s = 0
    for s in S:
        sum_over_s += min([env.distance_array_reachtimes[s, z] for z in Z])
    return sum_over_s
#enddef


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

if __name__ == "__main__":

    # gridsizefull_array = [7]
    # H_array = [4]
    #
    # acc_dict = {}

    env_args = {
        "gridsizefull": 10,
        "R_max": 10,
        "gamma": 0.95,
        "terminalState": 1,
        "randomMoveProb": 0.0,
    }

    env = Environment(env_args)
    Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, env.reward)
    # print(env.wall_vertical)
    # print(env.wall_horizontal)
    print(env.gate_states)
    env.draw(V, pi_s, env.reward, True, strname="some", fignum=3)
    exit(0)

    delta_s_array = get_delta_s_given_policy(env, pi_d, pi_s, tol=1e-10)
    upper_c_for_delt_s = sorted(delta_s_array)[len(delta_s_array)//2]

    active_states = np.concatenate((env.gate_states, env.goal_state))

    H_set = range(10, 21)

    obj_value, R_sol = reward_design_model_based_smooth(env, pi_d=pi_d, pi_s=pi_s, R_max=10, H_set=H_set,
                                                 s_active=active_states,
                                                 delta_s_array=delta_s_array, C_1=1, C_2=0,
                                                 dict_s_opt_actions_arr={},
                                                 state_only_reward_flag=False,
                                                 is_delta_s_const=True,
                                                 upper_c_for_delt_s=upper_c_for_delt_s, tol=1e-10)


    env.plot_reward(R_sol, 7, False, "max")
    exit(0)

    obj_value_sum, R_sol_sum = reward_design_model_based_sum(env, pi_d=pi_d, pi_s=pi_s, R_max=10, H_set=H_set,
                                                 s_active=active_states,
                                                 delta_s_array=delta_s_array, C_1=1, C_2=0,
                                                 dict_s_opt_actions_arr={},
                                                 state_only_reward_flag=False,
                                                 is_delta_s_const=True,
                                                 upper_c_for_delt_s=upper_c_for_delt_s, tol=1e-10)

    env.plot_reward(R_sol_sum, 7, True, "sum")


    exit(0)
    print(env.gate_states)
    Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, env.reward)
    env.draw(V, pi_s, env.reward, False, strname="some", fignum=3)
    env.plot_reward(env.reward, 7, True)
    exit(0)
    # print(env.T[-1,:,3])
    # # print(env.T[0,:,0])
    # # print(env.T[0,:,0])
    #
    print(env.gate_states)
    Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, env.reward)
    # env.draw(V, pi_s, env.reward, True, strname="some", fignum=3)
    # exit(0)
    pi_d[-2] = 0
    # print(pi_d)
    # exit(0)

    for s in range(env.n_states):
        for a in range(env.n_actions):
            print(sum(env.T[s,:, a]))

    # Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, env.reward)
    # env.draw(V, pi_s, env.reward, True, strname="some", fignum=3)
    # exit()
    # print(env.distance_array[0, -1])

    # Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, env.reward)
    # T_reach = MDPSolver.calc_reachtimes_with_T_pi(env, pi_d)
    pi_s_new = np.ones((env.n_states, env.n_actions)) / env.n_actions
    # print(pi_s_new)
    # exit(0)
    T_reach_new = MDPSolver.calc_reachtimes_with_T_pi(env, pi_s_new)
    # print(T_reach[0, 12])
    # for s in range(env.n_states-1):
    #     print("s=", s, " ", np.round(T_reach_new[0, s], 4))
    # print()
    # exit(0)

    # exit(0)
    # exit(0)
    # exit(0)
    Z = env.goal_state

    for i in range(5):
        min_s = np.inf
        s_choosen = None
        for s in range(env.n_states):
            candidate = np.concatenate((Z, [s]))
            fclt = facility_function(env, np.arange(env.n_states-env.terminal_state), candidate)
            if min_s > fclt:
                min_s = fclt
                s_choosen = copy.deepcopy(s)
        Z = np.concatenate((Z, [s_choosen]))
    print("Z=", Z)
    print(env.gate_states)
    print("=======")

    for z in Z:
        print(z%env.gridsizefull, z//env.gridsizefull)
    print("============")
    for w in env.gate_states:
        print(w%env.gridsizefull, w//env.gridsizefull)

    # Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, env.reward)
    env.draw(V, pi_s, env.reward, True, strname="some", fignum=3)
    # exit(0)
    # exit(0)
    H_set = range(10, 21)
    incremental_randmark_state_array = Z


    delta_s_array = get_delta_s_given_policy(env, pi_d, pi_s, tol=1e-10)

    obj_value, R_sol = reward_design_model_based(env, pi_d=pi_d, pi_s=pi_s, R_max=10, H_set=H_set,
                                                 s_active=incremental_randmark_state_array,
                                                 delta_s_array=delta_s_array, C_1=1, C_2=0,
                                                 dict_s_opt_actions_arr={},
                                                 state_only_reward_flag=False,
                                                 is_delta_s_const=True,
                                                 upper_c_for_delt_s=2, tol=1e-10)

    env.plot_reward(R_sol, 7, True)

    _, _, pi_d_res, _ = MDPSolver.valueIteration(env, R_sol)
    print("pi_d_res", pi_d_res)
    print("pi_d", pi_d)
    print(np.linalg.norm(pi_d-pi_d_res))

    exit(0)






    exit(0)
    print(env.distance_fun(0, 12))
    print(env.distance_array[0, 13])
    print(env.distance_array[0, 12])
    exit(0)
    print(env.distance_array[6, 7])
    print(env.distance_array[10, 0])
    print(env.distance_array)
    print(np.all(np.abs(env.distance_array-env.distance_array.T) <=0))
    Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, env.reward)
    # env.plot_reward(env.reward, fignum=1, show=False)
    env.draw(V, pi_s, env.reward, True, strname="some", fignum=3)
    exit(0)
    state = 0
    Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, env.reward)
    # env.plot_reward(env.reward, fignum=1, show=False)
    env.draw(V, pi_s, env.reward, True, strname="some", fignum=3)
    # print(Q[112])
    exit(0)
    print(Q)
    print(env.n_states)
    exit(0)
    poss_actions = env.get_possible_actions_within_grid(state)
    next_states = env.get_next_states(state, poss_actions)
    print(next_states)
    print(env.T[state, :, poss_actions[0]])
    print(env.T[state, :, poss_actions[1]])
    # exit(0)
    print(env.T[0, :, 1])
    # exit(0)
    print(env.wall_horizontal)
    print(env.wall_vertical)
    print(env.get_possible_actions_within_grid(44))
    print(env.get_possible_actions_within_grid(45))
    Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, env.reward)
    # print(env.reward)
    print(pi_d.reshape(env.gridsizefull, env.gridsizefull))
    print(pi_d)
    print(pi_s)
    # print(Q)
    # print(env.reward)

    exit(0)
    for s in range(env.n_states):
        for a in range(env.n_actions):
            print(sum(env.T[s, :, a]))
    # print(env.T[13])
    print()
    exit(0)
    print(env.T[3, :, 0])
    print(env.T[3, :, 1])
    print(env.T[3, :, 2])
    print(env.T[3, :, 3])
    print(env.wall_horizontal)
    print(env.wall_vertical)
    exit(0)
        # print()

    Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, env.reward)
    print(Q[9])
    print(np.argmax(Q[9]))
    env.plot_reward(env.reward, 7, True)
    exit(0)
    exit(0)
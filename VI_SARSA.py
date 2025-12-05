import sys, time, argparse
import gym
import numpy as np
from tqdm import tqdm
from lib.common_utils import TabularUtils
from lib.regEnvs import *



class Tabular_DP:
    def __init__(self, args):
        self.env = args.env # 
        self.gamma = 0.99 # discount factor
        self.theta = 1e-5 # convergence threshold
        self.max_iterations = 1000
        self.nA = self.env.action_space.n # 4
        self.nS = self.env.observation_space.n # 16 or 64

    def compute_q_value_cur_state(self, s, value_func):
        q_s = np.zeros(self.nA)
        
        # return: q_value for state s [float array with shape (nA)]
        for a in range(self.nA):
            for prob, s_next, reward, done in self.env.P[s][a]:
                q_s[a] += prob * (reward + self.gamma * value_func[s_next])
        return q_s

    def value_iteration(self):
        value_func = np.zeros(self.nS)
        policy_optimal = np.zeros([self.nS, self.nA])
        
        # return1 V_optimal [float array with shape (nS)]
        # return2 policy_optimal [one hot array with shape (nS x nA)]
        for _ in range(self.max_iterations):
            delta = 0
            for s in range(self.nS):
                v_prev = value_func[s]
                q_s = self.compute_q_value_cur_state(s, value_func)
                value_func[s] = np.max(q_s)
                delta = max(delta, abs(v_prev - value_func[s]))
            if delta < self.theta:
                break
        for s in range(self.nS):
            q_s = self.compute_q_value_cur_state(s, value_func)
            policy_optimal[s, np.argmax(q_s)] = 1.0
        return value_func, policy_optimal



class Tabular_TD:
    def __init__(self, args):
        self.env = args.env
        self.num_episodes=10000
        self.gamma = 0.99
        self.alpha = 0.05
        self.env_nA = self.env.action_space.n
        self.env_nS = self.env.observation_space.n
        self.tabularUtils = TabularUtils(self.env)
    

    def sarsa(self, epsilon=0.2):
        Q = np.zeros((self.env_nS, self.env_nA))
        salsa_policy = np.zeros((self.env_nS, self.env_nA))
        # return1 Q value [float array with shape (nS x nA)]
        # return2 salsa_policy [one hot array with shape (nS x nA)]
        for _ in range(self.num_episodes):
            s = self.env.reset()
            a = self.tabularUtils.epsilon_greedy_policy(Q[s])
            done = False
            
            while not done:
                s_next, reward, done, _ = self.env.step(a)
                a_next = self.tabularUtils.epsilon_greedy_policy(Q[s_next])
                Q[s, a] += self.alpha * (reward + self.gamma * Q[s_next, a_next] - Q[s, a])
                s = s_next
                a = a_next

        salsa_policy = self.tabularUtils.Q_value_to_greedy_policy(Q)
        return Q, salsa_policy


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_size', dest='map_size', type=int, default=4,  # Default map size is 4x4
                        choices=[4, 8], help="Specify the map size: 4 or 8.")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    if args.map_size == 4:
        args.env_name = "FrozenLake-Deterministic-v1"
    elif args.map_size == 8:
        args.env_name = "FrozenLake-Deterministic-8x8-v1"
    args.env = gym.make(args.env_name)
    tabularUtils = TabularUtils(args.env)
    
    # example dummy policies
    if args.map_size == 4:
        dummy_policy = np.array([1, 2, 2, 1, 1, 0, 3, 1, 2, 1, 3, 1, 2, 2, 3, 0])
    elif args.map_size == 8:
        dummy_policy = np.array([2, 2, 2, 2, 2, 2, 2, 1,
                                 0, 0, 0, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0])
    one_not_dummy_policy = tabularUtils.deterministic_policy_to_onehot_policy(dummy_policy)    
    # render
    tabularUtils.render(one_not_dummy_policy)

    # test value iteration
    dp = Tabular_DP(args)
    value_func, policy_optimal = dp.value_iteration()
    print("Value Function:")
    print(value_func.reshape(args.map_size, args.map_size))
    
    # test SARSA
    td = Tabular_TD(args)
    # Q, policy_sarsa = td.sarsa()
    # Q, policy_sarsa = td.sarsa(epsilon=0.01)
    Q, policy_sarsa = td.sarsa(epsilon=0.2)
    print("Policy:")
    print(np.argmax(policy_sarsa, axis=1).reshape(args.map_size, args.map_size))

    # render a video
    tabularUtils.render(policy_optimal) # from Tabular_DP
    tabularUtils.render(policy_sarsa) # from Tabular_TD



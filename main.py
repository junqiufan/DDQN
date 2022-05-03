import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import rl_utils
from DDQN import DQN
from train_DQN import train_DQN


lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 50
buffer_size = 5000
minimal_size = 1000
batch_size = 64

env_name = 'Pendulum-v1'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = 11  # 将连续动作分成11个离散动作
# cuda
if  torch.cuda.is_available():
    print("choose to use gpu...")
    device = torch.device("cuda:0")
else:
    print("choose to use cpu...")
    device = torch.device("cpu")

random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)#DQN算法
    # agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
    #             target_update, device, 'DoubleDQN') #DDQN算法

    return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                              replay_buffer, minimal_size,
                                              batch_size)

    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    # plt.title('Double DQN on {}'.format(env_name))
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('DQN on {}'.format(env_name))
    #plt.title('Double DQN on {}'.format(env_name))
    plt.show()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助

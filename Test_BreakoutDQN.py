import time
import gym
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import Breakout_DQN as breakout
import matplotlib.pyplot as plt

# For windows OS please open the comments codes below
'''physical_devices = tf.config.list_physical_devices('GPU')
   tf.config.experimental.set_memory_growth(physical_devices[0], True)
   tf.device('device:XLA_GPU:0')'''


def choose_action(state, epsilon):
    """ Choose the next action based on the given state.

    :param state: the given state
    :param epsilon: the value of epsilon
    :return: action
    """
    state = np.reshape(state, (1, 84, 84, 4))
    if np.random.uniform() < epsilon:
        return np.random.choice(env.action_space.n)
    else:
        return np.argmax(model.predict(state)[0])


def test_model():
    """ Test our trained agent.

    :return: the rewards of each episode
    """
    reward_list = []
    print("Now test the model player:")
    for i in range(10):
        score = 0
        # env.render()
        obs = copy.deepcopy(env.reset())
        obs = breakout.preprocess(obs)
        s = np.stack([obs] * 4, axis=0).T
        while True:
            # env.render()
            a = choose_action(s, 0.1)
            obs_, reward, is_done, _ = env.step(a)
            # get the state after taking action
            obs_ = breakout.preprocess(obs_)
            res_s = s[:, :, :-1]
            obs_ = np.array(obs_)
            s_ = np.insert(res_s, 0, values=obs_, axis=2)
            score += reward
            s = s_
            if is_done:
                print('score:', score)
                reward_list.append(score)
                break
    env.close()
    print("Average reward for test: %s"%np.mean(reward_list))
    print("Max reward for test: %s"%np.max(reward_list))
    return reward_list

def test_random():
    """ Test the random agent.

    :return: the rewards of each episode
    """
    reward_list = []
    print("Now test the random player:")
    for i in range(10):
        score = 0
        env.reset()
        while True:
            # env.render()
            a = np.random.randint(env.action_space.n)
            _, reward, is_done, _ = env.step(a)
            score += reward
            if is_done:
                print('score:', score)
                reward_list.append(score)
                break
    env.close()
    print("Average reward for test: %s"%np.mean(reward_list))
    print("Max reward for test: %s"%np.max(reward_list))
    return reward_list

if __name__ == '__main__':
    # setup the game
    env = gym.make('BreakoutDeterministic-v4')
    # load model
    model = keras.models.load_model('breakout_model14000.h5')

    model_reward = test_model()
    random_reward = test_random()

    # visualize the results
    plt.plot(model_reward, label='dqn')
    plt.plot(random_reward, label='random')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('The test rewards for each episode')
    plt.legend()
    plt.savefig('./test_rewards.png')


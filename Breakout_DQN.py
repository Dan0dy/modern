from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import os
import time
from PIL import Image
from tensorflow import keras

# For windows OS please open the comments codes below
'''physical_devices = tf.config.list_physical_devices('GPU')
   tf.config.experimental.set_memory_growth(physical_devices[0], True)'''


def preprocess(img):
    """ Preprocess the frame, convert into greyscale and normalize.

    :param img: the frame
    :return: the frame after preprocessing
    """
    # convert rgb to grey
    img = img.mean(axis=2)
    # down sample 210*160 to 84*84
    img = np.array(Image.fromarray(img).resize((84, 84))).astype(np.uint8)
    # normalize
    img = img / 255.0
    return img


class DQN(object):

    def __init__(self, state_size, action_size, memory_size, batch_size):
        self.step = 0  # record the step for target model updating
        self.state_size = state_size  # (84, 84, 4)
        self.action_size = action_size  # 4
        self.update_freq = 1000
        self.tao = 0.001  # for soft update
        self.factor = 0.99  # gamma
        self.model = self.create_model()  # the actual model
        self.target_model = self.create_model()  # the target model
        # replay buffer
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replay_memory = RingBuf(memory_size)
        # the old replay buffer use deque
        # self.replay_memory = deque(
        #     maxlen=self.memory_size)  # replay_memory--> state \ action \ next_state \ reward \ is_done

    def create_model(self):
        """ Creat the model(the updated nn model).

        :return: the model
        """
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(8, 8), strides=4,
                                      activation='relu',
                                      input_shape=self.state_size))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2,
                                      activation='relu',
                                      input_shape=self.state_size))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                                      activation='relu',
                                       input_shape=self.state_size))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=512, activation='relu'))
        model.add(keras.layers.Dense(units=self.action_size, activation='linear'))
        model.compile(loss=keras.losses.Huber(), optimizer='rmsprop')
        return model

    def act(self, state, epsilon):
        """ Choose the action for each state.

        :param state: the state
        :param epsilon: the epsilon value of epsilon-greedy algorithm
        :return: the suggested action
        """
        state = np.reshape(state, (1, 84, 84, 4))
        if np.random.uniform() < epsilon:
            # random action
            return np.random.choice(self.action_size)
        else:
            # model predict action
            return np.argmax(self.model.predict(state)[0])

    def soft_update(self):
        """ The soft update implementation for target model updating.

        :return: None
        """
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        w_list = []
        for i in range(len(weights)):
            w = self.tao * weights[i] + (1.0 - self.tao) * target_weights[i]
            w_list.append(w)
        self.target_model.set_weights(w_list)

    def train(self):
        """ The main learning component of Q learning.

        :return: None
        """
        if (self.step + 1) % self.update_freq == 0:
            s_batch, next_s_batch, d_batch, a_batch = self.sample()
            one_hot_a_batch = keras.utils.to_categorical(a_batch, self.action_size)
            Q = self.model.predict(s_batch)
            Q_next = self.target_model.predict(next_s_batch)
            Q_next[d_batch] = 0  # if done, make the reward = 0
            Q_target = reward + self.factor * np.max(Q_next, axis=1)
            history = self.model.fit(s_batch, one_hot_a_batch * Q_target[:, None], verbose=0)
            # print(history.history['loss'][0])
            # update target model
            self.soft_update()
        self.step += 1

    def remember(self, state, action, reward, next_state, is_done):
        """ Record the replay history.

        :param state: the state of 4 continuing frames
        :param action: the action
        :param reward: the rewards earned after action
        :param next_state: next state
        :param is_done: if the game is done
        :return:
        """
        self.replay_memory.append((state, action, reward, next_state, is_done))

    def sample(self):
        """ Random sample in replay history.

        :return: the sample batch of state, next state, is done, action
        """
        # get the index of random sampled batch
        idx = random.sample(range(0, self.replay_memory.__len__() - 1), self.batch_size)
        # get the replay batch according to index
        replay_batch = np.stack(self.replay_memory.__getitem__(i) for i in idx)
        s_batch = np.array([replay[0] for replay in replay_batch])  # state
        next_s_batch = np.array([replay[3] for replay in replay_batch])  # next state
        d_batch = np.array([replay[4] for replay in replay_batch])  # is done(bool)
        a_batch = np.array([replay[1] for replay in replay_batch])  # action
        return s_batch, next_s_batch, d_batch, a_batch

    def save_model(self, file_path):
        """ Save the model

        :param file_path: the model name and save path
        :return: None
        """
        print('model saved')
        self.model.save(file_path)


class RingBuf:
    """
    Considering the running time, we adopted the replay buffer implementation in
    <Beat Atari with Deep Reinforcement Learning! (Part 1: DQN)>
    https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
    """

    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == '__main__':

    env = gym.make('BreakoutDeterministic-v4')
    action_size = env.action_space.n  # 4
    state_size = (84, 84, 4)
    # google_path = '/content/drive/My Drive/RL_A3_results/'

    episodes = 30000
    episode_size = 40000
    save_iter = 2000
    # replay buffer
    memory_size = 50000
    batch_size = 32
    replay_start_size = 5000
    # epsilon-greedy
    ini_epsilon = 1.0
    end_epsilon = 0.1
    epsilon_decay = (1.0 - 0.1) / 1000000

    agent = DQN(state_size, action_size, memory_size, batch_size)
    start_time = time.time()
    epsilon = ini_epsilon
    total_scores = []  # total scores for each episode
    total_steps = 0

    for epi in range(episodes):
        # save the model and rewards during training
        if (epi + 1) % save_iter == 0:
            if not os.path.exists('./iter_models'):
                os.mkdir('./iter_models')
            agent.save_model(f'./iter_models/breakout_model{epi + 1}.h5')
            plt.plot(total_scores, color='blue')
            plt.xlabel('Episode')
            plt.ylabel('Rewards')
            plt.title('The total rewards for each episode during training')
            plt.legend()
            plt.savefig(f'./iter_models/rewards{epi + 1}.png')

        obs = env.reset()
        obs = preprocess(obs)
        # stack 4 frames as 1 state
        state = np.stack([obs] * 4, axis=0).T

        epi_score = 0

        for t in range(episode_size):
            action = agent.act(state, epsilon)
            obs_, reward, is_done, _ = env.step(action)
            # get the state after taking action
            obs_ = preprocess(obs_)
            res_s = state[:, :, :-1]
            obs_ = np.array(obs_)
            next_state = np.insert(res_s, 0, values=obs_, axis=2)
            # save the memory
            agent.remember(state, action, reward, next_state, is_done)
            total_steps += 1
            # start to learn after filled replay buffer start size
            if total_steps > replay_start_size:
                agent.train()
                # annealing epsilon
                epsilon = max(end_epsilon, epsilon * (1 - epsilon_decay))
            state = next_state
            epi_score += reward
            if is_done:
                break

        total_scores.append(epi_score)
        print("Episode {} with Reward : {} at epsilon {}"
              .format(epi + 1, epi_score, epsilon))

    agent.save_model('./breakout_model_weights.h5')
    end_time = time.time()
    print("Finished!")
    print("Time passed: %s h %s min." % ((end_time - start_time) // 3600, ((end_time - start_time) % 3600) * 60))
    print("Max score obtained:", np.max(total_scores))

    # Show plot
    plt.plot(total_scores, color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('The total rewards for each episode during training')
    plt.legend()
    plt.savefig('./final_rewards.png')
    # plt.show()

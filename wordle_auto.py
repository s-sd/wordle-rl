import numpy as np
import os


words_list_path = r'/home/s-sd/Desktop/wordle_auto/word_list.txt'

f = open(words_list_path, 'r')

words_list = list(f.readlines())
words_list = [word.strip('1234567890,. \n .\' -') for word in words_list]

wordle_word_list = [word for word in words_list if len(word)==5]
wordle_word_list = [word for word in wordle_word_list if not any(ele.isupper() for ele in word)]


################################################
# Wordle
################################################

# 6 guesses
# 5 letter word
# if letter in word and position incorrect = +5
# if letter in word and position correct = +10
# if word correct = +100
# if word incorrect after 5 = -100

guess_word = 'flabb'

word = 'abbey'

board = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],])


def wordle_obs_from_words(board, word, guess_word, turn_num):
    obs = np.array([0, 0, 0, 0, 0])
    for ind, (i, j) in enumerate(zip(word, guess_word)):
        if i == j:
            obs[ind] = 1
        else:
            if j in word:
                obs[ind] = -1
    board[turn_num] = obs
    return board

def wordle_score_guess(word, guess_word):
    score = 0
    for i, j in zip(word, guess_word):
        if j in word:
            score += 5
        if i == j:
            score += 10
    if word == guess_word:
        score = 100
    return score


################################################
# Gym env
################################################

import gym
from gym import spaces

class WordleEnv(gym.Env):
    
    def __init__(self, wordle_word_list):
        
        
        self.board = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],])
        
        self.wordle_word_list = wordle_word_list
        self.word = 'aaaaa'
        self.turn_num = 0
        
        self.action_space = spaces.Discrete(len(self.wordle_word_list))
        self.observation_space = spaces.Box(low=-1, high=1, shape=np.shape(board))
        
        self.img_shape = (6, 5)
        
        self.last_action = 0
                
    def reset(self):
        self.board = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],])
        word_ind = np.random.randint(0, len(wordle_word_list))
        self.word = self.wordle_word_list[word_ind]
        self.turn_num = 0
        return self.board
    
    def step(self, action):
        guess_word_ind = action
        self.guess_word = self.wordle_word_list[guess_word_ind]
        
        self.board = wordle_obs_from_words(self.board, self.word, self.guess_word, self.turn_num)
        
        score = wordle_score_guess(self.word, self.guess_word)
        
        self.turn_num += 1
        
        if self.turn_num >= 6:
            done = True
        else:
            done = False
            
        if action == self.last_action:
            score -= 100
             
        self.last_action = action
        
        return self.board, score, done, {}
    
    def render(self, mode='human'):
        print(self.word)
        print(self.guess_word)
        
wordle = WordleEnv(wordle_word_list)

wordle.reset()

# wordle.word

# wordle.step(200)


################################################
# Training
################################################

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv

# def make_env():
#     return WordleEnv(wordle_word_list)

# wordle = DummyVecEnv([make_env])

# model = PPO("MlpPolicy", wordle, verbose=1)
# model.learn(total_timesteps=6*100)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = wordle


nb_actions = env.action_space.n
obs_dim = env.observation_space.shape

# Option 1 : Simple model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())



print(model.summary())


# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
mean_rewards_list = []

total_trials = 20

import matplotlib.pyplot as plt


for trial in range(total_trials):
    
    print(f'Trial {trial} / {total_trials}')
    
    dqn.fit(env, nb_steps=6*100, visualize=False, verbose=0)
    
    history = dqn.test(env, nb_episodes=5, visualize=False, verbose=0)
    mean_reward = np.mean(history.history['episode_reward'])
    
    mean_rewards_list += [mean_reward]
    
plt.plot(mean_rewards_list)

dqn.test(env, nb_episodes=10, visualize=True, verbose=2)

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import gym
from a2c import A2C
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    inputs = env.observation_space.shape[0]
    output = env.action_space.n
    agent = A2C(alpha=0.0001, beta=0.0005, gamma=0.99, input_dim=inputs,
                hidden_dim_1=32, hidden_dim_2=16, output_dim=output)
    reward_history = []
    num_episodes = 1000
    for i in range(num_episodes):
        done = False
        state = env.reset()
        episode_reward = 0
        while not done:
           # if i > 150 and i % 5 == 0:
            #    env.render()
            #env.render()
            action = agent.get_action(state)
            new_state, reward, done, _ = env.step(action)
            agent.train(state, action, reward, new_state, done)
            state = new_state
            episode_reward += reward
        print('episode: ', i, ' score:  %.2f' % episode_reward)
        reward_history.append(episode_reward)
    plt.plot(reward_history)
    plt.show()

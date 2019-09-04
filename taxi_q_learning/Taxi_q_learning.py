import gym
import numpy as np
import pickle
from tqdm import tqdm
env = gym.make("Taxi-v2")
epsilon = 1
epsilon_decay = 0.99998
q_table = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.float)
lr = 0.1
discount = 0.9
episode = 50000
print(env.reward_range)
def learn():
    comp = 0
    global epsilon, epsilon_decay
    for i in tqdm(range(episode)):
        state = env.reset()
        for j in range(100):
            if np.random.uniform(low=0, high=1) > epsilon:
                action = np.argmax(q_table[state, :])
            else:
                action = env.action_space.sample()
            new_state, reward, done, _ = env.step(action)
            q_table[state, action] = q_table[state, action] + lr * (reward + discount * np.max(q_table[new_state, :]) - q_table[state, action])
            state = new_state
            if done == True:
                break
        epsilon = max(0.1, epsilon * epsilon_decay)
        if i % 1000 == 0:
            print(epsilon, "comp", comp)
            comp = 0
    with open("q_table_q_l.plk", 'wb') as f:
        pickle.dump(q_table, f, 1)

def test():
    with open("q_table_q_l.plk", 'rb') as f:
        q_table = pickle.load(f)
    s = env.reset()
    print(q_table)
    to = 0
    done = False
    print(q_table)
    while not done:
        env.render()
        action = np.argmax(q_table[s, :])
        s, r, done, _ = env.step(action)
        print(r)
        to += r
    print(to)

if __name__ == "__main__":
    learn()
    test()





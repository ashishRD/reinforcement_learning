# -*- coding: utf-8 -*-

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

class A2C:
    def __init__(self, input_dim, output_dim, alpha=0.0001, beta=0.0005, gamma=0.99, hidden_dim_1=512, hidden_dim_2=256, save_model_loc='models/'):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        self.save_models_loc = save_model_loc
        self.action_space = [i for i in range(self.output_dim)]
        self.actor, self.critic, self.policy = self.build_nets()
        
    def build_nets(self):
        input = Input(shape=[self.input_dim, ])
        delta = Input(shape=[1, ])
        hidden_1 = Dense(units=self.hidden_dim_1, activation='relu')(input)
        hidden_2 = Dense(units=self.hidden_dim_2, activation='relu')(hidden_1)
        actor_pred = Dense(units=self.output_dim, activation='softmax')(hidden_2)
        critic_eval = Dense(units=1, activation='linear')(hidden_2)
        
        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)
            return K.sum(-log_lik*delta) 
        actor = Model(input=[input, delta], output=[actor_pred])
        actor.compile(optimizer=Adam(self.alpha), loss=custom_loss)
        critic = Model(input=[input], output=[critic_eval])
        critic.compile(optimizer=Adam(self.beta), loss='mean_squared_error')
        policy = Model(input=[input], output=[actor_pred])
        return actor, critic, policy
    def get_action(self, state):
        state = state[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        return action
    def train(self, state, action, reward, new_state, done):
        state = state[np.newaxis,:]
        new_state = new_state[np.newaxis,:]
        new_state_val = self.critic.predict(new_state)
        state_val = self.critic.predict(state)
        
        target = reward + self.gamma * new_state_val * (1-int(done))
        delta = target - state_val

        actions = np.zeros([1, self.output_dim])
        actions[np.arange(1), action] = 1
        self.actor.fit([state, delta], actions, verbose=0) 
        self.critic.fit(state, target, verbose=0)

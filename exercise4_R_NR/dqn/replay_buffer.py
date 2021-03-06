from collections import namedtuple
import numpy as np
import os
import gzip
import pickle
from utils import *

def preprocess_data(data, history_length=0):
    for i in range(len(data["state"])):
        # Save history
        # TODO history length
        image_hist = []
        for j in range(history_length+1):

            s = data["state"][i]
            s = state_preprocessing(s)
            image_hist.extend([s] * (history_length + 1))
            s = np.array(s).reshape(96, 96, 1)

            a = data["action"][i]
            a = action_to_id(a)

            ns = data["next_state"][i]
            ns = state_preprocessing(ns)
            ns = np.array(ns).reshape(96, 96, 1)

            data["state"][i] = s
            data["action"][i] = a
            data["next_state"][i] = ns
    return data

def store_data(data, datasets_dir="./data", filename='data.pkl.gzip'):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, filename)
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)


def load_data(history_length, path="./data"):
    data_file = os.path.join(path, "data.pkl.gzip")

    f = gzip.open(data_file, "rb")
    data = pickle.load(f)

    #data = preprocess_data(data)
    #store_data(data)

    if history_length != 0:
        data = preprocess_data(data, history_length)
        store_data(data, filename='data_{}.pkl.gzip'.format(history_length))

    #print(data["state"][1].shape)

    return data

class ReplayBuffer:

    # TODO: implement a capacity for the replay buffer (FIFO, capacity: 1e5 - 1e6)

    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, history_length=0, capacity=1e5, load_data=False):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])
        self.capacity = capacity

        if load_data == True:
            data = load_data(history_length)
            self._data.states.extend(data["state"])
            self._data.actions.extend(data["action"])
            self._data.next_states.extend(data["next_state"])
            self._data.rewards.extend(data["reward"])
            self._data.dones.extend(data["terminal"])

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        buffer_length = len(self._data.states)

        if buffer_length == self.capacity:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.dones.pop(0)

            self._data.states.append(state)
            self._data.actions.append(action)
            self._data.next_states.append(next_state)
            self._data.rewards.append(reward)
            self._data.dones.append(done)
        else:
            self._data.states.append(state)
            self._data.actions.append(action)
            self._data.next_states.append(next_state)
            self._data.rewards.append(reward)
            self._data.dones.append(done)

    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_dones = np.array([self._data.dones[i] for i in batch_indices])
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones

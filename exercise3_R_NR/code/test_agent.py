from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

from model import Model
from utils import *
'''
num_filters = 16
batch_size = 64
lr = 0.0001
history_length = 1

'''
n_minibatches = 1000
num_filters = 74
#batch_size = 64
batch_size = 32
lr = 0.00032224967019634816
history_length = 1 + 0


def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    state_hist = np.zeros((1, state.shape[0], state.shape[1], history_length))

    while True:
        
        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        #    state = ...
        state = rgb2gray(state)
        state_hist[0, :, :, 0:history_length - 1] = state_hist[0, :, :, 1:]
        state_hist[0, :, :, -1] = state

        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        # a = ...

        a = agent.sess.run(agent.y_pred, feed_dict={agent.x_placeholder:state_hist})[0]
        a = id_to_action(a)

        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps:
            print('Reward: ', episode_reward)
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    agent = Model(lr=lr, num_filters=num_filters, batch_size=batch_size, history_length=history_length)
    agent.load("models/batch/agent_32.ckpt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        print(str(i) + '/ ' + str(n_test_episodes-1))
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    #fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = "results_run/test/results_bc_agent-%s.json" % 'batch_32'
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

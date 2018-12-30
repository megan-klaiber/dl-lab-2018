from __future__ import print_function

import gym
from dqn.dqn_agent import DQNAgent
from train_carracing import run_episode
from dqn.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

    history_length =  0

    #TODO: Define networks and load agent
    # ....

    state_dim = (96, 96)
    num_actions = 5
    batch_size = 128

    Q = CNN(state_dim=state_dim, num_actions=num_actions, hidden=256, lr=0.0003, history_length=0)
    Q_target = CNNTargetNetwork(state_dim=state_dim, num_actions=num_actions, hidden=256, lr=0.0003, history_length=0)
    agent = DQNAgent(Q, Q_target, num_actions, discount_factor=0.99, batch_size=batch_size, game='carracing')
    agent.load("./models_carracing/dqn_agent.ckpt")

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)
        print(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')


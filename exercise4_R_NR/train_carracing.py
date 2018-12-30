# export DISPLAY=:0 

import sys
sys.path.append("../") 

import numpy as np
import gym
from dqn.dqn_agent import DQNAgent
from dqn.networks import CNN, CNNTargetNetwork
from tensorboard_evaluation import *
import itertools as it
from utils import *

def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)

    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)

        action_id = agent.act(state=state, deterministic=deterministic)
        action = id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_reward", "straight", "left", "right", "accel", "brake"])

    eval_reward = []


    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        if i < (num_episodes // 3):
            max_timesteps = 250
        else:
            max_timesteps = 1000

        stats = run_episode(env, agent, max_timesteps=max_timesteps, deterministic=False, do_training=True, rendering=False, skip_frames=4)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE)
                                                      })

        # TODO: evaluate agent with deterministic actions from time to time
        # ...
        if i % 20 == 0 and i > 0:
            e_reward = 0
            for i in range(5):
                eval = run_episode(env, agent, deterministic=True, do_training=False)
                e_reward += eval.episode_reward
            eval_reward.append(e_reward/5)


        if i % 100 == 0 or (i >= num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))

        print('reward: ', stats.episode_reward)

    path = os.path.join("./", "carracing_reward_eval_training")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "reward_eval_training_%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))

    fh = open(fname, "w")
    json.dump(eval_reward, fh)
    fh.close()

    print('eval reward: ', eval_reward)

    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    env = gym.make('CarRacing-v0').unwrapped

    state_dim = (96, 96)
    num_actions = 5
    num_episodes = 300
    batch_size = 128

    # TODO: Define Q network, target network and DQN agent
    # ...

    Q = CNN(state_dim=state_dim, num_actions=num_actions, hidden=256, lr=0.0003, history_length=0)
    Q_target = CNNTargetNetwork(state_dim=state_dim, num_actions=num_actions, hidden=256, lr=0.0003, history_length=0)
    agent = DQNAgent(Q, Q_target, num_actions, discount_factor=0.99, batch_size=batch_size, game='carracing')
    
    train_online(env, agent, num_episodes=num_episodes, history_length=0, model_dir="./models_carracing")


from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import argparse
import json

from model import Model
from utils import *
from tensorboard_evaluation import Evaluation

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    #print(len(data['state']))

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]

    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    # iterate over data to avoid memory error
    five_thsd_train = X_train.shape[0] // 5000
    five_thsd_valid = X_valid.shape[0] // 5000

    X_train_gray = np.zeros(X_train.shape[0:3])
    X_valid_gray = np.zeros(X_valid.shape[0:3])

    #X_train_stack= np.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2], history_length))
    #y_train_stack = np.zeros((y_train.shape[0], y_train.shape[1]))

    #X_valid_stack = np.zeros((X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], history_length))
    #y_valid_stack = np.zeros((y_valid.shape[0], y_valid.shape[1]))

    y_train_action_indices = {}
    for i in range(9):
        y_train_action_indices[i] = np.array([]).astype('int')

    for i in range(five_thsd_train):
        start = i * 5000
        end = (i+1) * 5000
        X_train_gray[start: end] = rgb2gray(X_train[start: end])

    for i in range(five_thsd_valid):
        start = i * 5000
        end = (i + 1) * 5000

        X_valid_gray[start: end] = rgb2gray(X_valid[start: end])


    #X_train_gray = X_train_gray.reshape(X_train_gray.shape[0], 96, 96, 1)
    #X_valid_gray = X_valid_gray.reshape(X_valid_gray.shape[0], 96, 96, 1)

    y_train_id = np.zeros((y_train.shape[0]), dtype=int)
    for i in range(y_train.shape[0]):
        y_train_id[i] = action_to_id(y_train[i])
        y_train_action_indices[y_train_id[i]] = np.append(y_train_action_indices[y_train_id[i]], i)

    y_valid_id = np.zeros((y_valid.shape[0]), dtype=int)
    for i in range(y_valid.shape[0]):
        y_valid_id[i] = action_to_id(y_valid[i])

    classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    return X_train_gray, one_hot(y_train_id, classes=classes), y_train_action_indices , X_valid_gray, one_hot(y_valid_id, classes=classes)

def sample_minibatch(X, y, batch_size, y_train_action_indices, history_length):
    X = X.reshape(X.shape[0], 96, 96)

    X_batch = np.zeros((batch_size, X.shape[1], X.shape[2], history_length))
    y_batch = np.zeros((batch_size, y.shape[1]))

    # iterate over indices / batch size
    for i in range(batch_size):
        actions = range(9)
        action = random.sample(actions,1)[0]

        indices = range(len(y_train_action_indices[action]) - history_length)
        #print(indices)
        index = random.sample(indices, 1)[0]

        idx = y_train_action_indices[action][index]

        #idx = random.randint(0, X.shape[0] - 1)

        for h in range(history_length):
            #print('1: ', X_batch[i, :, :, h].shape)
            #print('2: ', X[idx + h].shape)
            X_batch[i, :, :, h] = X[idx + h]
            y_batch[i] = y[idx + h]

    return X_batch, y_batch

def history(X, y, history_length):
    X_hist = np.zeros((X.shape[0]-history_length, X.shape[1], X.shape[2], history_length))
    y_hist = np.zeros((y.shape[0]-history_length, y.shape[1]))

    for h in range(history_length):
        for i in range(X.shape[0] - history_length):
            X_hist[i, :, :, h] = X[i + h]
            y_hist[i] = y[i + h]

    return X_hist, y_hist

def train_model(X_train, y_train, y_train_action_indices, X_valid, y_valid, n_minibatches, num_filters, batch_size, lr, history_length, optimizer='Adam', model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your neural network in model.py 
    agent = Model(lr=lr, num_filters=num_filters, batch_size=batch_size, history_length=history_length, optimizer=optimizer)

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training in your web browser
    # 
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     for i % 10 == 0:
    #         tensorboard_eval.write_episode_data(...)

    agent.sess.run(tf.global_variables_initializer())
    tensorboard_eval = Evaluation(tensorboard_dir)
    #tf.reset_default_graph()

    # accuracies
    train_loss = np.zeros((n_minibatches))
    train_accuracy = np.zeros((n_minibatches))
    train_error = np.zeros((n_minibatches))
    valid_accuracy = np.zeros((n_minibatches))
    valid_error = np.zeros((n_minibatches))

    # number of batches valid
    n_samples_valid = X_valid.shape[0]
    n_batches_valid = n_samples_valid // batch_size

    X_valid, y_valid = history(X_valid, y_valid, history_length)

    # iterate over n_minibatches
    for m in range(n_minibatches):

        X_batch, y_batch = sample_minibatch(X_train, y_train, batch_size, y_train_action_indices, history_length)

        # train
        _, temp_loss = agent.sess.run([agent.optimizer, agent.loss],
                                feed_dict={agent.x_placeholder: X_batch, agent.y_placeholder: y_batch})

        train_loss[m] = (temp_loss / batch_size)
        # train error / accuracy
        train_accuracy[m] = agent.accuracy.eval({agent.x_placeholder: X_batch, agent.y_placeholder: y_batch}, session = agent.sess)
        train_error[m] = 1 - train_accuracy[m]

        # validation error / accuracy
        # iterate to avoid memory error
        for b in range(n_batches_valid):
            # extracting a batch from x_train and y_train
            start = b * batch_size
            end = start + batch_size
            x_batch_valid = X_valid[start:end, ]
            y_batch_valid = y_valid[start:end, ]
            valid_accuracy[m] += agent.accuracy.eval({agent.x_placeholder: x_batch_valid, agent.y_placeholder: y_batch_valid}, session = agent.sess)

        valid_accuracy[m] = valid_accuracy[m] / n_batches_valid
        valid_error[m] = 1 - valid_accuracy[m]

        #X_valid_batch, y_valid_batch = sample_minibatch(X_valid, y_valid, batch_size, y_train_action_indices, history_length)
        #valid_accuracy[m] = agent.accuracy.eval({agent.x_placeholder: X_valid_batch, agent.y_placeholder: y_valid_batch}, session = agent.sess)
        #valid_error[m] = 1 - valid_accuracy[m]

        print("[%d/%d]: train_accuracy: %.4f, valid_accuracy: %.4f, valid_error: %.4f" % (
        m + 1, n_minibatches, train_accuracy[m] , valid_accuracy[m], valid_error[m]))

        eval_dict = {'loss': train_loss[m], 'train_err': train_error[m], 'train_acc': train_accuracy[m], 'valid_acc': valid_accuracy[m], 'valid_err': valid_error[m]}

        if m % 10 == 0:
            tensorboard_eval.write_episode_data(n_minibatches, eval_dict)


    # TODO: save your agent
    agent.save(os.path.join(model_dir, "agent_%i.ckpt" % (batch_size)))
    print("Model saved in file: %s" % model_dir)
    agent.sess.close()

    return train_loss, train_accuracy, train_error, valid_accuracy, valid_error

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--model_name", default='test', type=str, nargs="?",
                        help="Name of model.")
    args = parser.parse_args()

    model_name = args.model_name

    # n_minibatches = 1000
    # num_filters = 16
    # batch_size = 64
    # lr = 0.0001
    # history_length = 1

    n_minibatches = 1000
    num_filters = 74
    batch_size = 64
    lr = 0.00032224967019634816
    history_length = 0

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    #print(X_train.shape)

    # preprocess data
    X_train, y_train, y_train_action_indices, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=history_length+1)

    # train model (you can change the parameters!)
    train_loss, train_accuracy, train_error, valid_accuracy, valid_error = \
        train_model(X_train, y_train, y_train_action_indices, X_valid, y_valid, n_minibatches=n_minibatches, num_filters=num_filters, batch_size=batch_size, lr=lr, history_length=history_length+1)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results['n_minibatches'] = n_minibatches
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["lr"] = lr
    results['history_length'] = history_length

    results["train_loss"] = list(train_loss)
    results["train_accuracy"] = list(train_accuracy)
    results['train_error'] = list(train_error)
    results['valid_accuracy'] = list(valid_accuracy)
    results['valid_error'] = list(valid_error)

    path = os.path.join(args.output_path, "results_run")
    os.makedirs(path, exist_ok=True)

    model_name = str(batch_size)
    fname = os.path.join(path, "results_run_%s.json" % model_name)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()
 

#!/bin/bash

#learning_rates = [0.1,0.01,0.001,0.0001]

python3 cnn_mnist.py --learning_rate=0.1
python3 cnn_mnist.py --learning_rate=0.01
python3 cnn_mnist.py --learning_rate=0.001
python3 cnn_mnist.py --learning_rate=0.0001

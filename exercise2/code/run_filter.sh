#!/bin/bash

#filter_sizes = [1,3,5,7]

python3 cnn_mnist.py --learning_rate=0.1 --filter_size=1
python3 cnn_mnist.py --learning_rate=0.1 --filter_size=3
python3 cnn_mnist.py --learning_rate=0.1 --filter_size=5
python3 cnn_mnist.py --learning_rate=0.1 --filter_size=7

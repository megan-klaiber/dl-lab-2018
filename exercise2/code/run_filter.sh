#!/bin/bash

#filter_sizes = [1,3,5,7]

python3 cnn_mnist.py --filter_size=1 --model_name='1'
python3 cnn_mnist.py --filter_size=3 --model_name='3'
python3 cnn_mnist.py --filter_size=5 --model_name='5'
python3 cnn_mnist.py --filter_size=7 --model_name='7'

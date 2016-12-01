# Memory Network  

## Introduction ##
This demo provides a simple example usage of the external memory in a way similar to the Neural Turing Machine (NTM) with content based addressing and differentiable read and write head.
For more technical details, please refer to the [NTM paper](https://arxiv.org/abs/1410.5401).

## Task Description ##
Here we design a simple task for illustration purpose. The input is a sequence with variable number of zeros followed with a variable number of non-zero elements, e.g., [0, 0, 0, 3, 1, 5, ...]. The task is to memorize the first non-zero number (e.g., 3) and to output this number in the end after going through the whole sequence. 

## Folder Structure ##
* external_memory.py: the implementation of the external memory class. 
* external_memory_example.conf: example usage of the external memory class.
* data_provider_mem.py: generates the training and testing data for the example.
* train.sh and test.sh: the scripts to run training and testing.

## How to Run ##
* training: ./train.sh
* testing: ./test.sh




#!/bin/bash

set -e
cd `dirname $0`
export PYTHONPATH=$PWD/../../../../

configs=(test_fc layer_activations projections test_print_layer
test_sequence_pooling test_lstmemory_layer test_grumemory_layer
last_first_seq test_expand_layer test_ntm_layers test_hsigmoid
img_layers util_layers simple_rnn_layers unused_layers test_cost_layers
test_rnn_group beam_search)


for conf in ${configs[*]}
do
    python -m paddle.utils.dump_config $conf.py > $conf.protostr
done
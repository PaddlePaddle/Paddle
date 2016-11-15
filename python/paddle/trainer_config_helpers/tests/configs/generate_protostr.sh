#!/bin/bash

set -e
cd `dirname $0`
export PYTHONPATH=$PWD/../../../../

protostr=$PWD/protostr

configs=(test_fc layer_activations projections test_print_layer
test_sequence_pooling test_lstmemory_layer test_grumemory_layer
last_first_seq test_expand_layer test_ntm_layers test_hsigmoid
img_layers img_trans_layers util_layers simple_rnn_layers unused_layers test_cost_layers
test_rnn_group shared_fc shared_lstm test_cost_layers_with_weight
test_spp_layer test_bilinear_interp test_maxout test_bi_grumemory math_ops)

whole_configs=(test_split_datasource)

for conf in ${configs[*]}
do
    echo "Generating " $conf
    python -m paddle.utils.dump_config $conf.py > $protostr/$conf.protostr.unitest
done

for conf in ${whole_configs[*]}
do
    echo "Generating " $conf
    python -m paddle.utils.dump_config $conf.py "" --whole > $protostr/$conf.protostr.unitest
done

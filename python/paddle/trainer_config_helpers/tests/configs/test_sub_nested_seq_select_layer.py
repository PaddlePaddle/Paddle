#!/usr/bin/env python
#coding=utf-8
from paddle.trainer_config_helpers import *

beam_size = 5

data = data_layer(name='input_seq', size=300)
selected_ids = data_layer(name='input', size=beam_size)
sub_nest_seq = sub_nested_seq_layer(input=data, selected_indices=selected_ids)

outputs(sub_nest_seq)

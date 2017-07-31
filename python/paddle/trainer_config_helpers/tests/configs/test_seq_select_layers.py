#!/usr/bin/env python
#coding=utf-8
from paddle.trainer_config_helpers import *

data = data_layer(name='input', size=300)
prob = fc_layer(input=data, size=1, act=SequenceSoftmaxActivation())
sub_nest_seq = sub_nested_seq_layer(input=[data, prob], top_k=1)

outputs(sub_nest_seq)

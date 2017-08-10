#!/usr/bin/env python
#coding=utf-8
from paddle.trainer_config_helpers import *

data = data_layer(name='input', size=300)

data = data_layer(name="data", size=128)
scores = fc_layer(input=data, size=1, act=ExpActivation())
kmax_seq_id = kmax_sequence_score_layer(input=scores, beam_size=5)

outputs(kmax_seq_id)

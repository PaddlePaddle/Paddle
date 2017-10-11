#!/usr/bin/env python
#coding=utf-8
from paddle.trainer_config_helpers import *

data = data_layer(name="input_seq", size=128)
scores = fc_layer(input=data, size=1, act=ExpActivation())
kmax_seq_id = kmax_seq_score_layer(input=scores, beam_size=5)

outputs(kmax_seq_id)

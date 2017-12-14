#!/usr/bin/env python
#coding=utf-8
from paddle.trainer_config_helpers import *

input_seq = data_layer("word", size=128)
starts = data_layer("starts", size=5)
ends = data_layer("ends", size=5)

seq_slice1 = seq_slice_layer(input=input_seq, starts=starts, ends=ends)
seq_slice2 = seq_slice_layer(input=input_seq, starts=starts, ends=None)
seq_slice3 = seq_slice_layer(input=input_seq, starts=None, ends=ends)

outputs(seq_slice1, seq_slice2, seq_slice3)

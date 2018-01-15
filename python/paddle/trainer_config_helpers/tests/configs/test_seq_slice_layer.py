#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
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

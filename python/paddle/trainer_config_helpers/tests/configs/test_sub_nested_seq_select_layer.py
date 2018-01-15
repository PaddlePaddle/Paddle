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

beam_size = 5

data = data_layer(name='input_seq', size=300)
selected_ids = data_layer(name='input', size=beam_size)
sub_nest_seq = sub_nested_seq_layer(input=data, selected_indices=selected_ids)

outputs(sub_nest_seq)

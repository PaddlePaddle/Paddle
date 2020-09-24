# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy as np


def convert_params_for_cell(np_cell, paddle_cell):
    state = np_cell.parameters
    for k, v in paddle_cell.named_parameters():
        v.set_value(state[k])


def convert_params_for_cell_static(np_cell, paddle_cell, place):
    state = np_cell.parameters
    for k, v in paddle_cell.named_parameters():
        scope = paddle.static.global_scope()
        tensor = scope.find_var(v.name).get_tensor()
        tensor.set(state[k], place)


def convert_params_for_net(np_net, paddle_net):
    for np_layer, paddle_layer in zip(np_net, paddle_net):
        if hasattr(np_layer, "cell"):
            convert_params_for_cell(np_layer.cell, paddle_layer.cell)
        else:
            convert_params_for_cell(np_layer.cell_fw, paddle_layer.cell_fw)
            convert_params_for_cell(np_layer.cell_bw, paddle_layer.cell_bw)


def convert_params_for_net_static(np_net, paddle_net, place):
    for np_layer, paddle_layer in zip(np_net, paddle_net):
        if hasattr(np_layer, "cell"):
            convert_params_for_cell_static(np_layer.cell, paddle_layer.cell,
                                           place)
        else:
            convert_params_for_cell_static(np_layer.cell_fw,
                                           paddle_layer.cell_fw, place)
            convert_params_for_cell_static(np_layer.cell_bw,
                                           paddle_layer.cell_bw, place)

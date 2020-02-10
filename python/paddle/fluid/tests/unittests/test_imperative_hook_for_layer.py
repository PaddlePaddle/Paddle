# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import contextlib
import unittest
import numpy as np
import six

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.dygraph.base as base

from test_imperative_lod_tensor_to_selected_rows import SimpleNet


def forward_hook(layer, input, output):
    print(layer)
    print("input: ", type(input))
    print("output: ", type(output))
    print("print input")
    for val in input:
        print("input val:", val.shape)
    print("print output")
    if isinstance(output, tuple):
        for out_val in output:
            print("output val:", out_val.shape)
    else:
        print("output val:", output.shape)


def forward_pre_hook(layer, input):
    print(layer)
    print("forward_pre input: ", type(input))
    print("print input")
    for val in input:
        print("input val:", val.shape)


class Test_Forward_Hook(unittest.TestCase):
    def test_forward_hook(self):
        seed = 90

        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            with fluid.dygraph.guard(place):
                fluid.default_startup_program().random_seed = seed
                fluid.default_main_program().random_seed = seed
                backward_strategy = fluid.dygraph.BackwardStrategy()
                backward_strategy.sort_sum_gradient = True

                input_word = np.array(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7,
                     8]).reshape(6, 3).astype('int64')
                input_word = input_word.reshape((-1, 3, 1))
                y_data = np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8,
                     9]).reshape(6, 3).astype('int64')
                y_data = y_data.reshape((-1, 1))

                input = base.to_variable(input_word)
                y = base.to_variable(y_data)

                simplenet = SimpleNet(
                    hidden_size=20,
                    vocab_size=32,
                    num_steps=3,
                    init_scale=0.1,
                    is_sparse=False,
                    dtype="float32")
                adam = fluid.optimizer.SGDOptimizer(
                    learning_rate=0.001, parameter_list=simplenet.parameters())

                forward_hook_handle = simplenet.register_forward_hook(
                    forward_hook)
                forward_pre_hook_handle = simplenet.register_forward_pre_hook(
                    forward_pre_hook)
                print("register ok")
                outs = simplenet(input, y)
                print("simplenet with register", type(outs))
                forward_hook_handle.remove()
                forward_pre_hook_handle.remove()
                print("delete ok")
                outs = simplenet(input, y)
                print("simplenet without register", type(outs))


if __name__ == '__main__':
    unittest.main()

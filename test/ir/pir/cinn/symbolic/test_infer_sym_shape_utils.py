# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np

import paddle


def get_sym_shape_str_for_op(net, input_spec, op_name='builtin.shadow_output'):
    forward_program = net.forward.get_concrete_program(*input_spec)[
        1
    ].infer_program.forward_program
    all_sym_shape_str = []
    for op in forward_program.global_block().ops:
        if op.name() == op_name:
            all_sym_shape_str.append(op.attrs()['sym_shape_str'])

    return all_sym_shape_str


def check_infer_results(net, input_spec, op_name, expecteds):
    sym_shape_str_list = get_sym_shape_str_for_op(net, input_spec, op_name)

    np.testing.assert_equal(len(sym_shape_str_list), len(expecteds))
    for i in range(len(sym_shape_str_list)):
        np.testing.assert_equal(
            sym_shape_str_list[i].find(expecteds[i]),
            0,
            f'in case i = {i},: output shape ({sym_shape_str_list[i]}) is not expected {(expecteds[i])}',
        )


class TestBase(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        pass

    def test_eval_symbolic(self):
        pass

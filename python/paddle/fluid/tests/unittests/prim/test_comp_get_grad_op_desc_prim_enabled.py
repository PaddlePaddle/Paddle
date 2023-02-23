# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid import core

core._set_prim_backward_enabled(True)

import parameterized as param

import paddle
from paddle.fluid import core, framework


@param.parameterized_class(
    (
        'fwd_type',
        'inputs',
        'outputs',
        'no_grad_var',
        'grad_sub_block',
        'desired_ops',
    ),
    (
        (
            'tanh',
            {'X': ['x']},
            {'Out': ['y']},
            set(),
            tuple(),
            (
                'pow',
                'fill_constant',
                'elementwise_mul',
                'fill_constant',
                'elementwise_add',
                'elementwise_mul',
            ),
        ),
        ('empty', {}, {'Out': ['y']}, set(), tuple(), tuple()),
    ),
)
class TestGetGradOpDescPrimEnabled(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.enable_static()
        block = framework.Block(framework.Program(), 0)
        block.append_op(
            type=cls.fwd_type,
            inputs={
                n: [block.create_var(name=v, stop_gradient=False) for v in vs]
                for n, vs in cls.inputs.items()
            },
            outputs={
                n: [block.create_var(name=v, stop_gradient=False) for v in vs]
                for n, vs in cls.outputs.items()
            },
        )
        cls.fwd = block.ops[0].desc

    @classmethod
    def tearDownClass(cls):
        paddle.disable_static()

    def test_get_grad_op_desc(self):
        actual = tuple(
            desc.type()
            for desc in core.get_grad_op_desc(
                self.fwd, self.no_grad_var, self.grad_sub_block
            )[0]
        )
        self.assertEquals(actual, self.desired_ops)
        core._set_prim_backward_enabled(False)


if __name__ == '__main__':
    unittest.main()

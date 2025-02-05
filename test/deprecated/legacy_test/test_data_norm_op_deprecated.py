#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""This is unit test of Test data_norm Op."""

import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import Program, program_guard

paddle.enable_static()


class TestDataNormOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            x2 = paddle.static.data(name='x2', shape=[-1, 3, 4], dtype="int32")
            # self.assertRaises(TypeError, base.data_norm, x2)
            paddle.static.nn.data_norm(
                input=x2, param_attr={}, enable_scale_and_shift=True
            )

            # Test input with dimension 1
            paddle.enable_static()
            x3 = paddle.static.data("", shape=[0], dtype="float32")
            self.assertRaises(ValueError, paddle.static.nn.data_norm, x3)

            # The size of input in data_norm should not be 0.
            def test_0_size():
                paddle.enable_static()
                x = paddle.static.data(name='x', shape=[0, 3], dtype='float32')
                out = paddle.static.nn.data_norm(x, slot_dim=1)
                cpu = base.core.CPUPlace()
                exe = base.Executor(cpu)
                exe.run(base.default_startup_program())
                test_program = base.default_main_program().clone(for_test=True)
                exe.run(
                    test_program,
                    fetch_list=out,
                    feed={'x': np.ones([0, 3]).astype('float32')},
                )

            self.assertRaises(ValueError, test_0_size)


if __name__ == '__main__':
    unittest.main()

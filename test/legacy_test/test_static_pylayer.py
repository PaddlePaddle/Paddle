# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import fluid
from paddle.fluid import core
from paddle.fluid.framework import Program, program_guard

np.random.seed(123)


class TestStatocPyLayerInputOutput(unittest.TestCase):
    def test_return_single_var(self):
        """
        pseudocode:

        y = 3 * x
        dx = 3 * dy
        """

        paddle.enable_static()

        def forward_fn(x):
            return 3 * x

        def backward_fn(dy):
            return 3 * dy

        main_program = Program()
        start_program = Program()
        with program_guard(main_program, start_program):
            data = paddle.static.data(name="X", shape=[1], dtype="float32")
            out = paddle.static.nn.static_pylayer(
                forward_fn, [data], backward_fn
            )

        place = (
            fluid.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )
        exe = fluid.Executor(place)
        x = np.array([2.0], dtype=np.float32)
        (ret,) = exe.run(main_program, feed={"X": x}, fetch_list=[out.name])
        np.testing.assert_allclose(
            np.asarray(ret), np.array([6.0], np.float32), rtol=1e-05
        )

    def test_return_0d_tensor(self):
        """
        pseudocode:

        y = 3 * x
        dx = 3 * dy
        """

        paddle.enable_static()

        def forward_fn(x):
            return 3 * x

        def backward_fn(dy):
            return 3 * dy

        main_program = Program()
        start_program = Program()
        with program_guard(main_program, start_program):
            data = paddle.full(shape=[], dtype='float32', fill_value=2.0)
            out = paddle.static.nn.static_pylayer(
                forward_fn, [data], backward_fn
            )

        place = (
            fluid.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )
        exe = fluid.Executor(place)
        (ret,) = exe.run(main_program, fetch_list=[out.name])
        np.testing.assert_allclose(
            np.asarray(ret), np.array(6.0, np.float32), rtol=1e-05
        )
        self.assertEqual(ret.shape, ())


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np

import paddle
from paddle import base, static
from paddle.cinn.common import DefaultHostTarget, DefaultNVGPUTarget, Float
from paddle.cinn.frontend import Computation, NetBuilder

assert len(sys.argv) == 3
enable_gpu = sys.argv.pop()
naive_model_dir = sys.argv.pop()


class TestNetBuilder(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def get_paddle_result(self, inputdata):
        paddle.enable_static()

        a = static.data(name='A', shape=[24, 56, 56], dtype='float32')
        b = static.data(name='B', shape=[24, 56, 56], dtype='float32')
        c = paddle.add(a, b)
        d = paddle.nn.initializer.NumpyArrayInitializer(
            np.array(inputdata[2]).reshape((144, 24, 1, 1)).astype('float32')
        )
        res = paddle.nn.Conv2D(
            in_channels=24,
            out_channels=144,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            weight_attr=d,
        )(c)

        exe = static.Executor(paddle.CPUPlace())
        exe.run(static.default_startup_program())

        x = np.array(inputdata[0]).reshape((1, 24, 56, 56)).astype("float32")
        y = np.array(inputdata[1]).reshape((1, 24, 56, 56)).astype("float32")
        output = exe.run(feed={"A": x, "B": y}, fetch_list=[res])
        return np.array(output)

    def test_build_and_compile(self):
        builder = NetBuilder("test_basic")
        a = builder.create_input(Float(32), (1, 24, 56, 56), "A")
        b = builder.create_input(Float(32), (1, 24, 56, 56), "B")
        c = builder.add(a, b)
        d = builder.create_input(Float(32), (144, 24, 1, 1), "D")
        e = builder.conv(c, d)

        computation = Computation.build_and_compile(self.target, builder)

        A_data = np.random.random([1, 24, 56, 56]).astype("float32")
        B_data = np.random.random([1, 24, 56, 56]).astype("float32")
        D_data = np.random.random([144, 24, 1, 1]).astype("float32")

        computation.get_tensor("A").from_numpy(A_data, self.target)
        computation.get_tensor("B").from_numpy(B_data, self.target)
        computation.get_tensor("D").from_numpy(D_data, self.target)

        computation.execute()

        e_tensor = computation.get_tensor(str(e))
        edata_cinn = e_tensor.numpy(self.target)

        edata_paddle = self.get_paddle_result([A_data, B_data, D_data])

        np.testing.assert_allclose(edata_cinn, edata_paddle, atol=1e-5)


class TestCompilePaddleModel(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def test_compile_paddle_model(self):
        A_shape = [4, 30]
        A_data = np.random.random(A_shape).astype("float32")
        computation = Computation.compile_paddle_model(
            self.target, naive_model_dir, ["A"], [A_shape], False
        )

        A_tensor = computation.get_tensor("A")
        A_tensor.from_numpy(A_data, self.target)

        computation.execute()

        out = computation.get_tensor("fc_0.tmp_2")
        res_cinn = out.numpy(self.target)

        config = base.core.AnalysisConfig(naive_model_dir)
        config.disable_gpu()
        config.switch_ir_optim(False)
        paddle_predictor = base.core.create_paddle_predictor(config)
        data = base.core.PaddleTensor(A_data)
        paddle_out = paddle_predictor.run([data])
        res_paddle = paddle_out[0].as_ndarray()

        np.testing.assert_allclose(res_cinn, res_paddle, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

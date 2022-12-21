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

import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


class TestInstanceNorm(unittest.TestCase):
    def test_error(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            places.append(fluid.CUDAPlace(0))
        for p in places:

            def error1d():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                instance_norm1d = paddle.nn.InstanceNorm1D(1)
                instance_norm1d(fluid.dygraph.to_variable(x_data_4))

            def error2d():
                x_data_3 = np.random.random(size=(2, 1, 3)).astype('float32')
                instance_norm2d = paddle.nn.InstanceNorm2D(1)
                instance_norm2d(fluid.dygraph.to_variable(x_data_3))

            def error3d():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                instance_norm3d = paddle.nn.InstanceNorm3D(1)
                instance_norm3d(fluid.dygraph.to_variable(x_data_4))

            def weight_bias_false():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                instance_norm3d = paddle.nn.InstanceNorm3D(
                    1, weight_attr=False, bias_attr=False
                )

            with fluid.dygraph.guard(p):
                weight_bias_false()
                self.assertRaises(ValueError, error1d)
                self.assertRaises(ValueError, error2d)
                self.assertRaises(ValueError, error3d)

    def test_dygraph(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            shape = [4, 10, 4, 4]

            def compute_v1(x):
                with fluid.dygraph.guard(p):
                    bn = paddle.nn.InstanceNorm2D(shape[1])
                    y = bn(fluid.dygraph.to_variable(x))
                return y.numpy()

            def compute_v2(x):
                with fluid.dygraph.guard(p):
                    bn = paddle.nn.InstanceNorm2D(shape[1])
                    y = bn(fluid.dygraph.to_variable(x))
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)

    def test_static(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            exe = fluid.Executor(p)
            shape = [4, 10, 16, 16]

            def compute_v1(x_np):
                with program_guard(Program(), Program()):
                    ins = paddle.nn.InstanceNorm2D(shape[1])
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = ins(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            def compute_v2(x_np):
                with program_guard(Program(), Program()):
                    ins = paddle.nn.InstanceNorm2D(shape[1])
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = ins(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()

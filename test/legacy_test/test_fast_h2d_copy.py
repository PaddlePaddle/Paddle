# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_device_place

import paddle


@unittest.skipIf(
    not paddle.core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestFastCPUCopy1(unittest.TestCase):
    def setUp(self):
        self.input_np_a = np.random.random((2048, 192 * 4)).astype(np.float32)
        self.input_np_b = np.random.random((128, 192, 2048)).astype(np.float32)
        self.input_dtype = 'float32'
        paddle.device.set_device("cpu")
        self.pd_cpu_tmp = paddle.to_tensor(self.input_np_a)
        paddle.device.set_device("gpu:0")
        self.pd_gpu_tmp = paddle.to_tensor(self.input_np_b)

    def check_dygraph_result(self, place):
        paddle.device.set_device("gpu:0")
        pd_cpu_b = self.pd_cpu_tmp.narrow(1, 0, 192)
        pd_cpu_b = pd_cpu_b.transpose([1, 0])
        pd_param = self.pd_gpu_tmp[3]
        pd_param.copy_(pd_cpu_b)

        np_cpu_b = self.input_np_a[:, 0:192].transpose(1, 0)
        np_gpu_param = self.input_np_b[3]
        np_gpu_param = np_cpu_b

        np.testing.assert_allclose(np_cpu_b, pd_cpu_b.numpy())
        np.testing.assert_allclose(np_gpu_param, pd_param.cpu().numpy())

    def test_dygraph(self):
        self.check_dygraph_result(place=get_device_place())


@unittest.skipIf(
    not paddle.core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestFastCPUCopy2(unittest.TestCase):
    def setUp(self):
        self.input_np_a = np.random.random((2048, 192 * 4)).astype(np.float32)
        self.input_np_b = np.random.random((128, 2048, 192)).astype(np.float32)
        self.input_dtype = 'float32'
        paddle.device.set_device("cpu")
        self.pd_cpu_tmp = paddle.to_tensor(self.input_np_a)
        paddle.device.set_device("gpu:0")
        self.pd_gpu_tmp = paddle.to_tensor(self.input_np_b)

    def check_dygraph_result(self, place):
        paddle.device.set_device("gpu:0")
        pd_cpu_b = self.pd_cpu_tmp.narrow(0, 0, 192)
        pd_cpu_b = pd_cpu_b.transpose([1, 0])
        pd_param = self.pd_gpu_tmp[3]

        pd_param.copy_(pd_cpu_b)

        np_cpu_b = self.input_np_a[0:192, :].transpose(1, 0)
        np_gpu_param = self.input_np_b[3]
        np_gpu_param[0:768, :] = np_cpu_b

        np.testing.assert_allclose(np_cpu_b, pd_cpu_b.numpy())
        np.testing.assert_allclose(np_gpu_param, pd_param.cpu().numpy())

    def test_dygraph(self):
        self.check_dygraph_result(place=get_device_place())


@unittest.skipIf(
    not paddle.core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestFastCPUCopy3(unittest.TestCase):
    def setUp(self):
        src_shape = [2, 2]
        tgt_shape = [2, 4]
        # self.input_np_a = np.random.random((2,2)).astype(np.float32)
        # self.input_np_b = np.random.random((2,4)).astype(np.float32)
        self.input_dtype = 'float32'
        paddle.device.set_device("cpu")
        self.src_cpu = paddle.ones(src_shape, dtype="float32")
        paddle.device.set_device("gpu:0")
        self.dst_gpu = paddle.zeros(tgt_shape, dtype="float32")

    def check_dygraph_result(self, place):
        paddle.device.set_device("gpu:0")
        tmp_dst_gpu = self.dst_gpu[..., :2]
        tmp_dst_gpu.copy_(self.src_cpu)
        tmo_dst_gpu1 = self.dst_gpu[..., 2:]
        tmo_dst_gpu1.copy_(self.src_cpu)
        np.testing.assert_allclose(self.dst_gpu.numpy(), 1.0)

    def test_dygraph(self):
        self.check_dygraph_result(place=get_device_place())


if __name__ == '__main__':
    unittest.main()

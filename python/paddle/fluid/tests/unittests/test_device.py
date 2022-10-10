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

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework


class TestStaticDeviceManage(unittest.TestCase):

    def _test_device(self, device_name, device_class):
        paddle.set_device(device_name)

        out1 = paddle.zeros(shape=[1, 3], dtype='float32')
        out2 = paddle.ones(shape=[1, 3], dtype='float32')
        out3 = paddle.concat(x=[out1, out2], axis=0)

        exe = paddle.static.Executor()
        exe.run(paddle.fluid.default_startup_program())
        res = exe.run(fetch_list=[out3])

        device = paddle.get_device()
        self.assertEqual(isinstance(exe.place, device_class), True)
        self.assertEqual(device, device_name)

    def test_cpu_device(self):
        self._test_device("cpu", core.CPUPlace)

    def test_gpu_device(self):
        if core.is_compiled_with_cuda():
            self._test_device("gpu:0", core.CUDAPlace)

    def test_xpu_device(self):
        if core.is_compiled_with_xpu():
            self._test_device("xpu:0", core.XPUPlace)

    def test_npu_device(self):
        if core.is_compiled_with_npu():
            self._test_device("npu:0", core.NPUPlace)


class TestImperativeDeviceManage(unittest.TestCase):

    def test_cpu(self):
        with fluid.dygraph.guard():
            paddle.set_device('cpu')
            out1 = paddle.zeros(shape=[1, 3], dtype='float32')
            out2 = paddle.ones(shape=[1, 3], dtype='float32')
            out3 = paddle.concat(x=[out1, out2], axis=0)
            device = paddle.get_device()
            self.assertEqual(
                isinstance(framework._current_expected_place(), core.CPUPlace),
                True)
            self.assertEqual(device, "cpu")

    def test_gpu(self):
        if core.is_compiled_with_cuda():
            with fluid.dygraph.guard():
                paddle.set_device('gpu:0')
                out1 = paddle.zeros(shape=[1, 3], dtype='float32')
                out2 = paddle.ones(shape=[1, 3], dtype='float32')
                out3 = paddle.concat(x=[out1, out2], axis=0)
                device = paddle.get_device()
                self.assertEqual(
                    isinstance(framework._current_expected_place(),
                               core.CUDAPlace), True)
                self.assertEqual(device, "gpu:0")

    def test_xpu(self):
        if core.is_compiled_with_xpu():
            with fluid.dygraph.guard():
                out = paddle.to_tensor([1, 2])
                device = paddle.get_device()
                self.assertEqual(
                    isinstance(framework._current_expected_place(),
                               core.XPUPlace), True)
                self.assertTrue(out.place.is_xpu_place())
                self.assertEqual(device, "xpu:0")

    def test_npu(self):
        if core.is_compiled_with_npu():
            with fluid.dygraph.guard():
                paddle.set_device('npu:0')
                out1 = paddle.zeros(shape=[1, 3], dtype='float32')
                out2 = paddle.ones(shape=[1, 3], dtype='float32')
                out3 = paddle.concat(x=[out1, out2], axis=0)
                device = paddle.get_device()
                self.assertEqual(
                    isinstance(framework._current_expected_place(),
                               core.NPUPlace), True)
                self.assertTrue(out1.place.is_npu_place())
                self.assertTrue(out2.place.is_npu_place())
                self.assertTrue(out3.place.is_npu_place())
                self.assertEqual(device, "npu:0")


if __name__ == '__main__':
    unittest.main()

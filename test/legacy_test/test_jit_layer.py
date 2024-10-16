#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import tempfile
import unittest

import numpy as np

import paddle
from paddle.base.framework import _dygraph_place_guard
from paddle.jit.layer import Layer
from paddle.static import InputSpec

sys.path.append("../dygraph_to_static")
from dygraph_to_static_utils import enable_to_static_guard

paddle.seed(1)


class Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc1 = paddle.nn.Linear(4, 4)
        self.fc2 = paddle.nn.Linear(4, 4)
        self._bias = 0.4

    @paddle.jit.to_static(
        input_spec=[InputSpec([None, 4], dtype='float32')], full_graph=True
    )
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = paddle.nn.functional.relu(out)
        out = paddle.mean(out)
        return out

    @paddle.jit.to_static(
        input_spec=[InputSpec([None, 4], dtype='float32')], full_graph=True
    )
    def infer(self, input):
        out = self.fc2(input)
        out = out + self._bias
        out = paddle.mean(out)
        return out


class TestMultiLoad(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multi_load(self):
        with paddle.pir_utils.OldIrGuard():
            paddle.disable_static()
            x = paddle.full([2, 4], 2)
            model = Net()
            with enable_to_static_guard(False):
                forward_out1 = model.forward(x)
                infer_out1 = model.infer(x)
            model_path = os.path.join(self.temp_dir.name, 'multi_program')
            paddle.jit.save(model, model_path, combine_params=True)
            place = paddle.CPUPlace()
            if paddle.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
            jit_layer = Layer()
            jit_layer.load(model_path, place)
            forward_out2 = jit_layer.forward(x)
            infer_out2 = jit_layer.infer(x)
            np.testing.assert_allclose(
                forward_out1, forward_out2[0], rtol=1e-05
            )
            np.testing.assert_allclose(infer_out1, infer_out2[0], rtol=1e-05)

    def test_multi_jit_load(self):
        x = paddle.full([2, 4], 2)
        model = Net()
        with enable_to_static_guard(False):
            forward_out1 = model.forward(x)
            infer_out1 = model.infer(x)
        model_path = os.path.join(self.temp_dir.name, 'multi_program')
        paddle.jit.save(model, model_path, combine_params=True)

        jit_layer = paddle.jit.load(model_path)
        forward_out2 = jit_layer.forward(x)
        infer_out2 = jit_layer.infer(x)
        np.testing.assert_allclose(forward_out1, forward_out2, rtol=1e-05)
        np.testing.assert_allclose(infer_out1, infer_out2, rtol=1e-05)


class SaveLinear(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(80, 80)

    @paddle.jit.to_static(
        input_spec=[InputSpec(shape=[None, 80], dtype='float32')],
        full_graph=True,
    )
    def forward(self, x):
        out = self.linear(x)
        return out


class TestMKLOutput(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mkl_output(self):
        with paddle.pir_utils.OldIrGuard():
            paddle.disable_static()
            with _dygraph_place_guard(place=paddle.CPUPlace()):
                net = SaveLinear()
                model_path = os.path.join(self.temp_dir.name, 'save_linear')
                paddle.jit.save(net, model_path, combine_params=True)

                layer = Layer()
                print("load ", model_path)
                layer.load(model_path, paddle.CPUPlace())
                x = paddle.ones([498, 80])
                out = layer.forward(x)
                out = paddle.unsqueeze(out[0], 0)
                np.testing.assert_equal(out.shape, [1, 498, 80])

    def test_mkl_jit_output(self):
        with _dygraph_place_guard(place=paddle.CPUPlace()):
            net = SaveLinear()
            x = paddle.ones([498, 80])
            orig_out = net.forward(x)
            model_path = os.path.join(self.temp_dir.name, 'save_linear')
            paddle.jit.save(net, model_path, combine_params=True)

            layer = paddle.jit.load(model_path)

            out = layer.forward(x)
            np.testing.assert_equal(
                np.mean(orig_out.numpy()), np.mean(out.numpy())
            )
            out = paddle.unsqueeze(out, 0)
            np.testing.assert_equal(out.shape, [1, 498, 80])


if __name__ == '__main__':
    unittest.main()

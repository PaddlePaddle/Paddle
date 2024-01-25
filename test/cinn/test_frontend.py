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

from paddle import base
from paddle.cinn.common import DefaultHostTarget, DefaultNVGPUTarget
from paddle.cinn.frontend import Interpreter

assert len(sys.argv) == 1 + 2 + 1  # model and enable_gpu count
enable_gpu = sys.argv.pop()
multi_fc_model_dir = sys.argv.pop()
naive_model_dir = sys.argv.pop()
""" class TestFrontend(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def paddle_verify(self, result):
        paddle.enable_static()

        a = static.data(name='A', shape=[24, 56, 56], dtype='float32')
        b = static.data(name='B', shape=[24, 56, 56], dtype='float32')
        c = paddle.add(a, b)
        d = paddle.nn.functional.relu(c)
        e = paddle.nn.initializer.NumpyArrayInitializer(
            np.array(result[2]).reshape((144, 24, 1, 1)).astype("float32"))
        f = static.nn.conv2d(
            input=d,
            num_filters=144,
            filter_size=1,
            stride=1,
            padding=0,
            dilation=1,
            param_attr=e)
        g = paddle.scale(f, scale=2.0, bias=0.5)
        res = paddle.nn.functional.softmax(g, axis=1)

        exe = static.Executor(paddle.CPUPlace())
        exe.run(static.default_startup_program())

        x = np.array(result[0]).reshape((1, 24, 56, 56)).astype("float32")
        y = np.array(result[1]).reshape((1, 24, 56, 56)).astype("float32")
        output = exe.run(feed={"A": x, "B": y}, fetch_list=[res])
        output = np.array(output).reshape(-1)
        print("result in paddle_verify: \n")
        for i in range(0, output.shape[0]):
            if np.abs(output[i] - result[len(result) - 1][i]) > 1e-4:
                print("Error! ", i, "-th data has diff with target data:\n",
                      output[i], " vs: ", result[len(result) - 1][i],
                      ". Diff is: ", output[i] - result[len(result) - 1][i])
        self.assertTrue(
            np.allclose(result[len(result) - 1], output, atol=1e-4))

    def test_basic(self):
        prog = Program()

        a = Variable("A").set_type(Float(32)).set_shape([1, 24, 56, 56])
        b = Variable("B").set_type(Float(32)).set_shape([1, 24, 56, 56])
        c = prog.add(a, b)
        d = prog.relu(c)
        e = Variable("E").set_type(Float(32)).set_shape([144, 24, 1, 1])
        f = prog.conv2d(d, e, {
            "stride": [1, 1],
            "dilation": [1, 1],
            "padding": [0, 0]
        })
        g = prog.scale(f, {"scale": 2.0, "bias": 0.5})
        h = prog.softmax(g, {"axis": 1})

        self.assertEqual(prog.size(), 5)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        tensor_data = [
            np.random.random([1, 24, 56, 56]).astype("float32"),
            np.random.random([1, 24, 56, 56]).astype("float32"),
            np.random.random([144, 24, 1, 1]).astype("float32")
        ]
        result = prog.build_and_get_output(self.target, [a, b, e], tensor_data,
                                           [h])
        result[0].set_type(Float(32))
        result = result[0].numpy(self.target).reshape(-1)
        tensor_data.append(result)
        self.paddle_verify(tensor_data) """


class TestLoadPaddleModel_FC(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

        self.model_dir = naive_model_dir

    def get_paddle_inference_result(self, model_dir, data):
        config = base.core.AnalysisConfig(model_dir)
        config.disable_gpu()
        config.switch_ir_optim(False)
        self.paddle_predictor = base.core.create_paddle_predictor(config)
        data = base.core.PaddleTensor(data)
        results = self.paddle_predictor.run([data])

        return results[0].as_ndarray()

    def test_model(self):
        np.random.seed(0)
        self.x_shape = [4, 30]
        x_data = (
            np.random.random(self.x_shape).astype("float16").astype("float32")
        )
        print('x_data', x_data)

        self.executor = Interpreter(["A"], [self.x_shape])
        self.executor.load_paddle_model(self.model_dir, self.target, False)
        a_t = self.executor.get_tensor("A")
        a_t.from_numpy(x_data, self.target)

        self.executor.run()

        out = self.executor.get_tensor("fc_0.tmp_2")
        target_data = self.get_paddle_inference_result(self.model_dir, x_data)
        print("target_data's shape is: ", target_data.shape)
        out_np = out.numpy(self.target)
        print("cinn data's shape is: ", out_np.shape)

        np.testing.assert_allclose(out_np, target_data, atol=1e-4)


class TestLoadPaddleModel_MultiFC(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

        self.model_dir = multi_fc_model_dir

    def get_paddle_inference_result(self, model_dir, data):
        config = base.core.AnalysisConfig(model_dir)
        config.disable_gpu()
        config.switch_ir_optim(False)
        self.paddle_predictor = base.core.create_paddle_predictor(config)
        data = base.core.PaddleTensor(data)
        results = self.paddle_predictor.run([data])

        return results[0].as_ndarray()

    def test_model(self):
        np.random.seed(0)
        self.x_shape = [8, 64]
        x_data = np.random.random(self.x_shape).astype("float32")

        self.executor = Interpreter(["A"], [self.x_shape])
        self.executor.load_paddle_model(self.model_dir, self.target, False)
        a_t = self.executor.get_tensor("A")
        a_t.from_numpy(x_data, self.target)

        self.executor.run()

        out = self.executor.get_tensor("fc_5.tmp_2")
        target = self.get_paddle_inference_result(self.model_dir, x_data)

        np.testing.assert_allclose(out.numpy(self.target), target, atol=1e-4)


if __name__ == "__main__":
    unittest.main()

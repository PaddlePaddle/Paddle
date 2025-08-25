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
from op_test import get_places

import paddle
from paddle import base


class TestSigmoidAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.places = get_places()
        self.init_data()

    def init_data(self):
        self.shape = [10, 15]
        self.dtype = "float32"
        self.np_input = np.random.uniform(-1, 1, self.shape).astype(self.dtype)

    def ref_forward(self, x):
        return 1 / (1 + np.exp(-x))

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        paddle_dygraph_out = []
        # Position args (args)
        out1 = paddle.sigmoid(x)
        paddle_dygraph_out.append(out1)
        # Key words args (kwargs) for paddle
        out2 = paddle.sigmoid(x=x)
        paddle_dygraph_out.append(out2)
        # Key words args for torch
        out3 = paddle.sigmoid(input=x)
        paddle_dygraph_out.append(out3)
        # Tensor method args
        out4 = x.sigmoid()
        paddle_dygraph_out.append(out4)
        # Test out
        out5 = paddle.empty([])
        paddle.sigmoid(x, out=out5)
        paddle_dygraph_out.append(out5)
        # Reference output
        ref_out = self.ref_forward(self.np_input)
        # Check
        for i in range(len(paddle_dygraph_out)):
            np.testing.assert_allclose(
                ref_out, paddle_dygraph_out[i].numpy(), rtol=1e-05
            )
        paddle.enable_static()

    def test_static_Compatibility(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            # Position args (args)
            out1 = paddle.sigmoid(x)
            # Key words args (kwargs) for paddle
            out2 = paddle.sigmoid(x=x)
            # Key words args for torch
            out3 = paddle.sigmoid(input=x)
            # Tensor method args
            out4 = x.sigmoid()
            exe = base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_input},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = self.ref_forward(self.np_input)
            for i in range(len(fetches)):
                np.testing.assert_allclose(fetches[i], ref_out, rtol=1e-05)


class TestTensorSigmoidAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.places = get_places()
        self.init_data()

    def init_data(self):
        self.shape = [10, 15]
        self.dtype = "float32"
        self.np_input = np.random.uniform(-1, 1, self.shape).astype(self.dtype)

    def ref_forward(self, x):
        return 1 / (1 + np.exp(-x))

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        paddle_dygraph_out = []
        # Position args (args)
        out1 = paddle.Tensor.sigmoid(x)
        paddle_dygraph_out.append(out1)
        # Key words args (kwargs) for paddle
        out2 = paddle.Tensor.sigmoid(x=x)
        paddle_dygraph_out.append(out2)
        # Key words args for torch
        out3 = paddle.Tensor.sigmoid(input=x)
        paddle_dygraph_out.append(out3)
        # Tensor method args
        out4 = x.sigmoid()
        paddle_dygraph_out.append(out4)
        # Test out
        out5 = paddle.empty([])
        paddle.Tensor.sigmoid(x, out=out5)
        paddle_dygraph_out.append(out5)
        # Reference output
        ref_out = self.ref_forward(self.np_input)
        # Check
        for i in range(len(paddle_dygraph_out)):
            np.testing.assert_allclose(
                ref_out, paddle_dygraph_out[i].numpy(), rtol=1e-05
            )
        paddle.enable_static()

    def test_static_Compatibility(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            # Position args (args)
            out1 = paddle.Tensor.sigmoid(x)
            # Key words args (kwargs) for paddle
            out2 = paddle.Tensor.sigmoid(x=x)
            # Key words args for torch
            out3 = paddle.Tensor.sigmoid(input=x)
            # Tensor method args
            out4 = x.sigmoid()
            exe = base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_input},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = self.ref_forward(self.np_input)
            for i in range(len(fetches)):
                np.testing.assert_allclose(fetches[i], ref_out, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()

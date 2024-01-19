# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_legacy_and_pt_and_pir,
)

import paddle
from paddle.framework import use_pir_api


class GradLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x.stop_gradient = False
        y = x * x
        dx = paddle.grad(outputs=[y], inputs=[x])[0]
        return dx


class GradLinearLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(5, 5, bias_attr=False)

    def forward(self, x):
        x.stop_gradient = False
        tmp = x + x
        for i in range(10):
            tmp = self.linear(tmp)
        out = tmp
        dx = paddle.grad(
            [out], [x], None, create_graph=True, allow_unused=False
        )[0]
        return dx


class NoGradLinearLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(5, 5, bias_attr=False)

    def forward(self, x):
        x.stop_gradient = False

        with paddle.no_grad():
            y = self.linear(x)

        out = y + x
        return out


class TestGrad(Dy2StTestBase):
    def setUp(self):
        self.func = GradLayer()
        self.x = paddle.ones(shape=[10, 2, 5], dtype='float32')
        self.x.stop_gradient = False

    @test_legacy_and_pt_and_pir
    def test_forward(self):
        dygraph_res = self.func(self.x).numpy()
        static_res = paddle.jit.to_static(self.func)(self.x).numpy()
        np.testing.assert_allclose(static_res, dygraph_res, rtol=1e-05)


class TestGradLinear(TestGrad):
    def setUp(self):
        self.func = GradLinearLayer()
        self.x = paddle.ones(shape=[10, 2, 5], dtype='float32')
        self.x.stop_gradient = False

        self.temp_dir = tempfile.TemporaryDirectory()
        self.infer_model_path = os.path.join(
            self.temp_dir.name, 'double_grad_infer_model'
        )
        self.train_model_path = os.path.join(
            self.temp_dir.name, 'double_grad_train_model'
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @test_legacy_and_pt_and_pir
    def test_save_infer_program(self):
        # TODO(pir-save-load): Fix this after we support save/load in PIR
        if use_pir_api():
            return
        static_fn = paddle.jit.to_static(self.func)
        input_spec = [
            paddle.static.InputSpec(shape=[10, 2, 5], dtype='float32')
        ]
        paddle.jit.save(static_fn, self.infer_model_path, input_spec=input_spec)
        load_func = paddle.jit.load(self.infer_model_path)

        origin_res = static_fn(self.x).numpy()
        load_res = load_func(self.x).numpy()
        np.testing.assert_allclose(origin_res, load_res, rtol=1e-05)

    @test_legacy_and_pt_and_pir
    def test_save_train_program(self):
        static_fn = paddle.jit.to_static(self.func)
        grad_clip = paddle.nn.ClipGradByGlobalNorm(2.0)
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01,
            grad_clip=grad_clip,
            parameters=static_fn.parameters(),
        )
        for i in range(10):
            out = static_fn(self.x)
            avg_loss = paddle.mean(paddle.abs(out - 1))
            avg_loss.backward()
            optimizer.minimize(avg_loss)

            static_fn.clear_gradients()

        # TODO(pir-save-load): Fix this after we support save/load in PIR
        if use_pir_api():
            return
        paddle.jit.save(static_fn, self.train_model_path)
        load_func = paddle.jit.load(self.train_model_path)

        origin_res = static_fn(self.x).numpy()
        load_res = load_func(self.x).numpy()
        np.testing.assert_allclose(origin_res, load_res, rtol=1e-05)


class TestNoGradLinear(TestGradLinear):
    def setUp(self):
        self.func = NoGradLinearLayer()
        self.x = paddle.ones(shape=[10, 2, 5], dtype='float32')
        self.x.stop_gradient = False

        self.temp_dir = tempfile.TemporaryDirectory()
        self.infer_model_path = os.path.join(
            self.temp_dir.name, 'no_grad_infer_model'
        )
        self.train_model_path = os.path.join(
            self.temp_dir.name, 'no_grad_train_model'
        )

    def tearDown(self):
        self.temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()

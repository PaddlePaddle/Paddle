# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


class TestAlias(unittest.TestCase):
    def setUp(self):
        self.autogradObject = paddle.autograd.PyLayerContext
        self.functionObject = paddle.autograd.function.FunctionCtx

    def test_compatibility(self):
        self.assertTrue(self.autogradObject is self.functionObject)


class TestMarkNonDifferentiableAlias(TestAlias):
    def setUp(self):
        self.autogradObject = (
            paddle.autograd.PyLayerContext.mark_non_differentiable
        )
        self.functionObject = (
            paddle.autograd.function.FunctionCtx.mark_non_differentiable
        )


class TestSaveForBackwardAlias(TestAlias):
    def setUp(self):
        self.autogradObject = paddle.autograd.PyLayerContext.save_for_backward
        self.functionObject = (
            paddle.autograd.function.FunctionCtx.save_for_backward
        )


class TestSavedTensorAlias(TestAlias):
    def setUp(self):
        self.autogradObject = paddle.autograd.PyLayerContext.saved_tensor
        self.functionObject = paddle.autograd.function.FunctionCtx.saved_tensor


class TestEnableGradAlias(TestAlias):
    def setUp(self):
        self.autogradObject = paddle.autograd.enable_grad
        self.functionObject = paddle.enable_grad

    def test_public_api(self):
        self.assertIn('enable_grad', paddle.autograd.__all__)

    def test_context_manager(self):
        x = paddle.to_tensor([1.0], stop_gradient=False)
        with paddle.no_grad():
            y = x * 2
            with paddle.autograd.enable_grad():
                z = x * 2

        self.assertTrue(y.stop_gradient)
        self.assertFalse(z.stop_gradient)

    def test_decorator(self):
        @paddle.autograd.enable_grad()
        def double(x):
            return x * 2

        x = paddle.to_tensor([1.0], stop_gradient=False)
        with paddle.no_grad():
            y = double(x)

        self.assertFalse(y.stop_gradient)


class TestSetMaterializeGradsAlias(TestAlias):
    def setUp(self):
        self.autogradObject = (
            paddle.autograd.PyLayerContext.set_materialize_grads
        )
        self.functionObject = (
            paddle.autograd.function.FunctionCtx.set_materialize_grads
        )


if __name__ == "__main__":
    unittest.main()

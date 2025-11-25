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

import paddle
from paddle.compat.proxy import create_fake_class, create_fake_function


def use_torch_inside_inner_function():
    import torch

    return torch.sin(torch.tensor([0.0, 1.0, 2.0])).numpy()


class TestTorchProxy(unittest.TestCase):
    def test_enable_torch_proxy(self):
        with self.assertRaises(ModuleNotFoundError):
            import torch

        paddle.compat.enable_torch_proxy()
        import torch

        self.assertIs(torch.sin, paddle.sin)

        import torch.nn

        self.assertIs(torch.nn.Conv2d, paddle.nn.Conv2d)

        import torch.nn.functional

        self.assertIs(torch.nn.functional.sigmoid, paddle.nn.functional.sigmoid)

        with self.assertRaises(ModuleNotFoundError):
            import torch.nonexistent_module

        paddle.compat.disable_torch_proxy()
        with self.assertRaises(ModuleNotFoundError):
            import torch
        with self.assertRaises(ModuleNotFoundError):
            import torch.nn
        with self.assertRaises(ModuleNotFoundError):
            import torch.nn.functional

    def test_use_torch_proxy_guard(self):
        with self.assertRaises(ModuleNotFoundError):
            import torch
        with paddle.compat.use_torch_proxy_guard():
            import torch

            self.assertIs(torch.sin, paddle.sin)
        with self.assertRaises(ModuleNotFoundError):
            import torch

        with paddle.compat.use_torch_proxy_guard():
            import torch

            self.assertIs(torch.cos, paddle.cos)
            with paddle.compat.use_torch_proxy_guard(enable=False):
                with self.assertRaises(ModuleNotFoundError):
                    import torch
                with paddle.compat.use_torch_proxy_guard(enable=True):
                    import torch

        with self.assertRaises(ModuleNotFoundError):
            import torch

    @paddle.compat.use_torch_proxy_guard()
    def test_use_torch_inside_inner_function(self):
        result = use_torch_inside_inner_function()

        np.testing.assert_allclose(
            result, np.sin([0.0, 1.0, 2.0]), atol=1e-6, rtol=1e-6
        )


class TestTorchProxyBlockedModule(unittest.TestCase):
    def test_blocked_module(self):
        with paddle.compat.use_torch_proxy_guard():
            with self.assertRaises(ModuleNotFoundError):
                import torch._dynamo.allow_in_graph  # noqa: F401

            with self.assertRaises(AttributeError):
                import torch_proxy_blocked_module

            paddle.compat.extend_torch_proxy_blocked_modules(
                {"torch_proxy_blocked_module"}
            )
            import torch_proxy_blocked_module

            # Use torch specific function out of execute module stage
            torch_proxy_blocked_module.use_torch_specific_fn()


class TestOverrideTorchModule(unittest.TestCase):
    @paddle.compat.use_torch_proxy_guard()
    def test_relu(self):
        import torch

        self.assertIs(torch.relu, paddle.nn.functional.relu)

    @paddle.compat.use_torch_proxy_guard()
    def test_access_compat_functions_by_getattr(self):
        import torch

        self.assertIs(torch.nn.Unfold, paddle.compat.nn.Unfold)
        self.assertIs(torch.nn.Linear, paddle.compat.nn.Linear)

    @paddle.compat.use_torch_proxy_guard()
    def test_access_compat_functions_by_import(self):
        from torch.nn.functional import linear, softmax

        self.assertIs(softmax, paddle.compat.nn.functional.softmax)
        self.assertIs(linear, paddle.compat.nn.functional.linear)


class TestFakeInterface(unittest.TestCase):
    def test_fake_interface(self):
        FakeGenerator = create_fake_class(
            "torch.Generator",
            {"manual_seed": create_fake_function("manual_seed")},
        )

        fake_gen = FakeGenerator()
        self.assertTrue(hasattr(fake_gen, "manual_seed"))


if __name__ == "__main__":
    unittest.main()

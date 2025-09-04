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

import paddle


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


if __name__ == "__main__":
    unittest.main()

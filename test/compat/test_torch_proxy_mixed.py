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

import pathlib
import sys
import unittest

import paddle
from paddle.compat.proxy import (
    ProxyModule,
)

sys.path.append(str(pathlib.Path(__file__).parent / "fake_modules"))
sys.path.append(str(pathlib.Path(__file__).parent / "fake_torch_modules"))


class TestTorchProxyMixRealTorch(unittest.TestCase):
    def check_is_not_proxy(self):
        import torch
        from torch.nn.functional import relu

        self.assertNotIsInstance(torch, ProxyModule)
        self.assertIsNot(paddle.nn.functional.relu, relu)

    def check_is_proxy(self):
        import torch
        from torch.nn.functional import relu

        self.assertIsInstance(torch, ProxyModule)
        self.assertIs(paddle.nn.functional.relu, relu)

    def test_nested_torch_proxy(self):
        self.check_is_not_proxy()
        with paddle.compat.use_torch_proxy_guard(enable=True):
            self.check_is_proxy()

            with paddle.compat.use_torch_proxy_guard(enable=False):
                self.check_is_not_proxy()

                with paddle.compat.use_torch_proxy_guard(enable=True):
                    self.check_is_proxy()
                    with paddle.compat.use_torch_proxy_guard(enable=True):
                        self.check_is_proxy()
                    with paddle.compat.use_torch_proxy_guard(enable=False):
                        self.check_is_not_proxy()

                self.check_is_not_proxy()

            self.check_is_proxy()

        self.check_is_not_proxy()

    def test_local_enabled_module_import(self):
        self.check_is_not_proxy()
        with paddle.compat.use_torch_proxy_guard(
            enable=True, scope={"torch_proxy_local_enabled_module"}
        ):
            self.check_is_not_proxy()
        self.check_is_not_proxy()

    def test_blocked_module_import(self):
        self.check_is_not_proxy()
        paddle.compat.extend_torch_proxy_blocked_modules(
            {"torch_proxy_blocked_module"}
        )
        with paddle.compat.use_torch_proxy_guard(enable=True):
            import torch_proxy_blocked_module  # noqa: F401

            self.check_is_proxy()
        self.check_is_not_proxy()

    def test_blocked_module_inside_local_enabled_proxy_import(self):
        self.check_is_not_proxy()
        paddle.compat.extend_torch_proxy_blocked_modules(
            {"torch_proxy_blocked_module"}
        )
        with paddle.compat.use_torch_proxy_guard(
            enable=True, scope={"torch_proxy_local_enabled_module"}
        ):
            import torch_proxy_blocked_module  # noqa: F401

            self.check_is_not_proxy()
        self.check_is_not_proxy()


if __name__ == "__main__":
    unittest.main()

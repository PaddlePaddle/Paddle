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
from paddle.base import core
from paddle.base.framework import flag_guard
from paddle.jit.dy2static.utils import Backend, backend_guard

CINN_FLAG_NAME = "FLAGS_use_cinn"
PRIM_DYNAMIC_FLAG_NAME = "FLAGS_prim_enable_dynamic"


class TestJitBackend(unittest.TestCase):
    def test_cinn_backend(self):
        # skip check if not compiled with CINN
        if not paddle.is_compiled_with_cinn():
            return
        with backend_guard(Backend.CINN):
            # Check all prim enabled
            self.assertTrue(core._is_fwd_prim_enabled())
            self.assertTrue(core._is_bwd_prim_enabled())
            # Check prim dynamic shape enabled
            self.assertTrue(core._enable_prim_dynamic_shape())
            # Check auto recompute enabled
            self.assertTrue(core._enable_auto_recompute())
            # Check CINN mode enabled and C++ flag has been switched
            self.assertTrue(paddle.base.framework.in_cinn_mode())
            self.assertTrue(paddle.get_flags(CINN_FLAG_NAME)[CINN_FLAG_NAME])

    def test_phi_backend(self):
        with backend_guard(Backend.PHI):
            # Check all prim disabled
            self.assertFalse(core._is_fwd_prim_enabled())
            self.assertFalse(core._is_bwd_prim_enabled())
            # Check prim dynamic shape disabled
            self.assertFalse(core._enable_prim_dynamic_shape())
            # Check auto recompute disabled
            self.assertFalse(core._enable_auto_recompute())
            # Check CINN mode disabled
            if (
                paddle.is_compiled_with_cinn()  # skip check if not compiled with CINN
            ):
                self.assertFalse(paddle.base.framework.in_cinn_mode())
                self.assertFalse(
                    paddle.get_flags(CINN_FLAG_NAME)[CINN_FLAG_NAME]
                )

    def test_phi_backend_with_outer_state(self):
        with (
            flag_guard(PRIM_DYNAMIC_FLAG_NAME, True),
            backend_guard(Backend.PHI),
        ):
            # In PHI backend, backend guard should not affect prim dynamic shape.
            # It will use the outer state
            self.assertTrue(core._enable_prim_dynamic_shape())


if __name__ == '__main__':
    unittest.main()

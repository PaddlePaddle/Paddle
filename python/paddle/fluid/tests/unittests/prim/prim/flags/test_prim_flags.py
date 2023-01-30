# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

from paddle.fluid import core


class TestPrimFlags(unittest.TestCase):
    def test_prim_flags(self):
        self.assertFalse(core._is_bwd_prim_enabled())
        self.assertFalse(core._is_fwd_prim_enabled())

        os.environ['FLAGS_prim_backward'] = "True"
        core.check_and_set_prim_all_enabled()
        self.assertTrue(core._is_bwd_prim_enabled())
        os.environ['FLAGS_prim_forward'] = "True"
        core.check_and_set_prim_all_enabled()
        self.assertTrue(core._is_fwd_prim_enabled())
        os.environ['FLAGS_prim_all'] = "False"
        core.check_and_set_prim_all_enabled()
        self.assertFalse(core._is_bwd_prim_enabled())
        self.assertFalse(core._is_fwd_prim_enabled())

        os.environ['FLAGS_prim_all'] = "True"
        core.check_and_set_prim_all_enabled()
        self.assertTrue(core._is_bwd_prim_enabled())
        self.assertTrue(core._is_fwd_prim_enabled())

        del os.environ['FLAGS_prim_all']
        os.environ['FLAGS_prim_backward'] = "False"
        core.check_and_set_prim_all_enabled()
        self.assertFalse(core._is_bwd_prim_enabled())
        os.environ['FLAGS_prim_forward'] = "False"
        core.check_and_set_prim_all_enabled()
        self.assertFalse(core._is_fwd_prim_enabled())


if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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

from __future__ import print_function

import unittest

import paddle
from paddle.fluid.dygraph.jit import _run_save_pre_hooks, _clear_save_pre_hooks, _register_save_pre_hook

_counter = 0


class TestPreSaveHooks(unittest.TestCase):
    def test_pre_save_hook_functions(self):
        def fake_func(*args, **kwgs):
            global _counter
            _counter += 1

        remove_handler = _register_save_pre_hook(fake_func)
        self.assertEqual(len(paddle.fluid.dygraph.jit._save_pre_hooks), 1)
        self.assertTrue(
            paddle.fluid.dygraph.jit._save_pre_hooks[0] is fake_func)

        # Test of avoiding redundancy hanging
        remove_handler = _register_save_pre_hook(fake_func)
        self.assertEqual(len(paddle.fluid.dygraph.jit._save_pre_hooks), 1)
        self.assertTrue(
            paddle.fluid.dygraph.jit._save_pre_hooks[0] is fake_func)

        remove_handler.remove()
        self.assertEqual(len(paddle.fluid.dygraph.jit._save_pre_hooks), 0)

        remove_handler = _register_save_pre_hook(fake_func)
        _clear_save_pre_hooks()
        self.assertEqual(len(paddle.fluid.dygraph.jit._save_pre_hooks), 0)

        global _counter
        _counter = 0
        remove_handler = _register_save_pre_hook(fake_func)
        func_with_hook = _run_save_pre_hooks(fake_func)
        func_with_hook(None, None)
        self.assertEqual(_counter, 2)


if __name__ == '__main__':
    unittest.main()

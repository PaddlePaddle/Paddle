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

import os
import platform
import shutil
import unittest

import paddle


# Test the debug_info_path params in backward
# Just check whether the debug file is generated
class TestDumpDebugInfo(unittest.TestCase):
    def test_dump_debug_info(self):
        # windows ci may have some permission issues
        if 'Windows' == platform.system():
            return
        paddle.disable_static()
        self._test_Tensor_backward()
        self._test_paddle_grad()
        self._test_autograd_backward()
        paddle.enable_static()

    def _test_Tensor_backward(self):
        x = paddle.randn([5, 5], dtype='float32')
        y = paddle.randn([5, 5], dtype='float16')
        x.stop_gradient = False
        y.stop_gradient = False

        z = x + y
        h = z + 1
        h = h * z
        w = h + y
        # test Tensor.backward
        debug_info_path = "_Tensor_backward/"
        w.backward(debug_info_path=debug_info_path)
        self._check_files_in_directory(debug_info_path)
        shutil.rmtree(debug_info_path)

    def _test_paddle_grad(self):
        x = paddle.randn([5, 5], dtype='float32')
        y = paddle.randn([5, 5], dtype='float32')
        x.stop_gradient = False
        y.stop_gradient = False
        z = x + y
        h = x * z
        w = h + y
        # test paddle.grad
        debug_info_path = "_paddle_grad/"
        grads = paddle.grad([w], [x, y], debug_info_path=debug_info_path)
        self._check_files_in_directory(debug_info_path)
        shutil.rmtree(debug_info_path)

    def _test_autograd_backward(self):
        x = paddle.randn([5, 5], dtype='float32')
        y = paddle.randn([5, 5], dtype='float32')
        x.stop_gradient = False
        y.stop_gradient = False
        z = x + y
        h = x * z
        w = h + y
        # test paddle.autograd.backward
        debug_info_path = "_paddle_autograd_backward/"
        grads = paddle.autograd.backward(
            [x, y], [None, None], debug_info_path=debug_info_path
        )
        self._check_files_in_directory(debug_info_path)
        shutil.rmtree(debug_info_path)

    def _check_files_in_directory(self, directory):
        # Check whether the expected file exists in the directory
        entries = os.listdir(directory)
        files = [
            entry
            for entry in entries
            if os.path.isfile(os.path.join(directory, entry))
        ]
        expect_keywards_in_file_name = [
            "backward_graph.dot",
            "ref_forward_graph.dot",
            "call_stack.log",
        ]
        for keywords in expect_keywards_in_file_name:
            if not any(keywords in f for f in files):
                raise AssertionError(
                    f"Error: File '{keywords}' not found in directory '{directory}'! "
                )


if __name__ == "__main__":
    unittest.main()

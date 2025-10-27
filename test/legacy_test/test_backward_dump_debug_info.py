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
import subprocess
import sys
import unittest
from unittest.mock import patch

import paddle


# Test the dump_backward_graph_path params in backward
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
        dump_backward_graph_path = "_Tensor_backward/"
        w.backward(dump_backward_graph_path=dump_backward_graph_path)
        self._check_files_in_directory(dump_backward_graph_path)
        shutil.rmtree(dump_backward_graph_path)

    def _test_paddle_grad(self):
        x = paddle.randn([5, 5], dtype='float32')
        y = paddle.randn([5, 5], dtype='float32')
        x.stop_gradient = False
        y.stop_gradient = False
        z = x + y
        h = x * z
        w = h + y
        # test paddle.grad
        dump_backward_graph_path = "_paddle_grad/"
        grads = paddle.grad(
            [w], [x, y], dump_backward_graph_path=dump_backward_graph_path
        )
        self._check_files_in_directory(dump_backward_graph_path)
        shutil.rmtree(dump_backward_graph_path)

    def _test_autograd_backward(self):
        x = paddle.randn([5, 5], dtype='float32')
        y = paddle.randn([5, 5], dtype='float32')
        x.stop_gradient = False
        y.stop_gradient = False
        z = x + y
        h = x * z
        w = h + y
        # test paddle.autograd.backward
        dump_backward_graph_path = "_paddle_autograd_backward/"
        grads = paddle.autograd.backward(
            [x, y],
            [None, None],
            dump_backward_graph_path=dump_backward_graph_path,
        )
        self._check_files_in_directory(dump_backward_graph_path)
        shutil.rmtree(dump_backward_graph_path)

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

    # Just execute vlog for the coverage ci
    def test_vlog(self):
        code = """
import os
os.environ['GLOG_v'] = '{glog_level}'
import paddle
x = paddle.randn([5, 5], dtype='float32')
y = paddle.randn([5, 5], dtype='float32')
x.stop_gradient = False
y.stop_gradient = False
z = x + y
h = x * z
w = h + y
grads = paddle.autograd.backward(
    [x, y],
    [None, None],
)
paddle.base.core.set_vlog_level(4)
        """
        process = subprocess.run(
            [sys.executable, '-c', code.format(glog_level=4)],
            capture_output=True,
            text=True,
        )
        process = subprocess.run(
            [sys.executable, '-c', code.format(glog_level=5)],
            capture_output=True,
            text=True,
        )
        process = subprocess.run(
            [sys.executable, '-c', code.format(glog_level=6)],
            capture_output=True,
            text=True,
        )
        process = subprocess.run(
            [sys.executable, '-c', code.format(glog_level=11)],
            capture_output=True,
            text=True,
        )

    def test_manual_vlog(self):
        if 'Windows' == platform.system():
            return
        code = """
import os
os.environ['GLOG_v'] = '6'
os.environ['FLAGS_dump_grad_node_forward_stack_path']="call_stack.log"
import paddle
import paddle.nn.functional as F
import paddle.nn as nn

paddle.base.core.set_vlog_level({"backward":6, "*": 7})

x = paddle.randn([3,3],dtype='float16')
y = paddle.randn([3,3],dtype='float32')
z = paddle.randn([3,3],dtype='float64')
w = paddle.randn([3,3],dtype='float64')
x.stop_gradient = False
y.stop_gradient = False
z.stop_gradient = False
w.stop_gradient = True

conv_x  = paddle.randn((2, 3, 8, 8), dtype='float32')
conv_w = paddle.randn((6, 3, 3, 3), dtype='float16')

sync_bn_input = paddle.to_tensor([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]).astype('float32')

conv_x.stop_gradient = False
conv_w.stop_gradient = False
sync_bn_input.stop_gradient = False

with paddle.amp.auto_cast(enable=True):
    out1 = paddle.add_n([x,y])
    out2 = paddle.multiply(x,y)
    out6 = F.conv2d(conv_x,conv_w)

out3 = paddle.add_n([out1,y])
out4 = paddle.multiply(out2,z)
out5 = paddle.multiply_(w, y)
if paddle.is_compiled_with_cuda():
    sync_batch_norm = nn.SyncBatchNorm(2)
    hidden1 = sync_batch_norm(sync_bn_input)
loss = out1 + out2 + out3 + out4 + out5 + out6.sum()+hidden1.sum()
loss.backward(dump_backward_graph_path="./backward")


    """
        process = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
        )

    # Test the input path is not valid
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_raise_not_a_directory_error(self, mock_isdir, mock_exists):
        # simulate
        mock_exists.return_value = True
        mock_isdir.return_value = False
        paddle.disable_static()
        with self.assertRaises(NotADirectoryError) as context:
            x = paddle.randn([5, 5], dtype='float32')
            y = paddle.randn([5, 5], dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False
            z = x + y
            h = x * z
            w = h + y
            grads = paddle.autograd.backward(
                [x, y], [None, None], dump_backward_graph_path="/path/to/check"
            )

        self.assertTrue(
            " path:'/path/to/check' must be directory "
            in str(context.exception)
        )

    @patch('os.makedirs')
    def test_create_file_error(self, mock_makedirs):
        # simulate os.makedirs throw exception
        mock_makedirs.side_effect = Exception("Mocked exception")
        with self.assertRaises(OSError) as context:
            x = paddle.randn([5, 5], dtype='float32')
            y = paddle.randn([5, 5], dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False
            z = x + y
            h = x * z
            w = h + y
            grads = paddle.autograd.backward(
                [x, y], [None, None], dump_backward_graph_path='/path/to/create'
            )

        self.assertTrue(
            "Create '/path/to/create' failed : Mocked exception"
            in str(context.exception)
        )


class TestSetVlogLevelError(unittest.TestCase):
    def test_input_invalid(self):
        with self.assertRaises(ValueError):
            paddle.base.core.set_vlog_level("3")


class TestVlogGuard(unittest.TestCase):
    # Just run it for coverage ci and don't check the res
    def test_guard(self):
        with paddle.base.framework.vlog_guard(0):
            x = paddle.randn([3, 3], dtype='float16')
        with paddle.base.framework.vlog_guard({"api": 0}):
            y = paddle.randn([3, 3], dtype='float16')

    # Check the invalid input
    def test_error(self):
        def test_invalid_input():
            with paddle.base.framework.vlog_guard("api"):
                x = paddle.randn([3, 3], dtype='float16')

        self.assertRaises(TypeError, test_invalid_input)


if __name__ == "__main__":
    unittest.main()

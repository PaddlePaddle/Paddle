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


class TestCaptureBackwardSubGraphGuard(unittest.TestCase):
    # Just run it for coverage ci and don't check the res
    def test_guard(self):
        # windows ci may have some permission issues
        if 'Windows' == platform.system() or not paddle.is_compiled_with_cuda():
            return
        import paddle.nn.functional as F
        from paddle import nn

        dump_dir_path = "_test/"

        x = paddle.randn([3, 3], dtype='float16')
        y = paddle.randn([3, 3], dtype='float32')
        z = paddle.randn([3, 3], dtype='float64')
        w = paddle.randn([3, 3], dtype='float64')
        x.stop_gradient = False
        y.stop_gradient = False
        z.stop_gradient = False
        w.stop_gradient = True
        y = y + y
        with paddle.base.framework.capture_backward_subgraph_guard(
            dump_dir_path, True
        ):
            conv_x = paddle.randn((2, 3, 8, 8), dtype='float32')
            conv_w = paddle.randn((6, 3, 3, 3), dtype='float16')

            sync_bn_input = paddle.to_tensor(
                [[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]
            ).astype('float32')

            conv_x.stop_gradient = False
            conv_w.stop_gradient = False
            sync_bn_input.stop_gradient = False
            with paddle.amp.auto_cast(enable=True):
                out1 = paddle.add_n([x, y])
                out2 = paddle.multiply(x, y)
                out6 = F.conv2d(conv_x, conv_w)

            out3 = paddle.add_n([out1, y])
            out4 = paddle.multiply(out2, z)
            out5 = paddle.multiply_(w, y)
            if paddle.is_compiled_with_cuda():
                sync_batch_norm = nn.SyncBatchNorm(2)
                hidden1 = sync_batch_norm(sync_bn_input)
            out7 = out6.sum() + hidden1.sum()
        loss = out1 + out2 + out3 + out4 + out5 + out7
        loss.backward()
        self._check_files_in_directory(dump_dir_path)
        shutil.rmtree(dump_dir_path)

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
            "grad_tensors.log",
        ]
        for keywords in expect_keywards_in_file_name:
            if not any(keywords in f for f in files):
                raise AssertionError(
                    f"Error: File '{keywords}' not found in directory '{directory}'! "
                )


if __name__ == "__main__":
    unittest.main()

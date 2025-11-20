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
import unittest

import paddle


class TestCaptureFwdGraph(unittest.TestCase):
    def test_capture_fwd_graph(self):
        # Do not test on Windows because it haven't graphviz exe on Windows CI
        if 'Windows' == platform.system():
            return
        x = paddle.rand([3, 9, 5])
        file_path = "./fwd_graph"
        with paddle.utils.capture_fwd_graph_guard(file_path):
            out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=1)
            y = out0 + out1
            z = out1 - out2
            w = out0 * out2
            out3 = paddle.concat([y, z, w], axis=1)

        if os.path.exists(file_path + ".svg"):
            file_size = os.path.getsize(file_path + ".svg")
            assert file_size > 0
        else:
            raise Exception("The graph file does not exist.")


if __name__ == "__main__":
    unittest.main()

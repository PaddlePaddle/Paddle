#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# Note:
# 0D Tensor indicates that the tensor's dimension is 0
# 0D Tensor's shape is always [], numel is 1
# which can be created by paddle.rand([])

import unittest

import paddle


class TestNoBackwardAPIStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.exe = paddle.static.Executor()

    def test_static_embedding(self):
        ids = paddle.full(shape=[], fill_value=1, dtype='int64')
        emb = paddle.static.nn.embedding(ids, (20, 3))
        prog = paddle.static.default_main_program()
        self.exe.run(paddle.static.default_startup_program())
        res = self.exe.run(prog, fetch_list=[emb])
        self.assertEqual(res[0].shape, (3,))


if __name__ == "__main__":
    unittest.main()

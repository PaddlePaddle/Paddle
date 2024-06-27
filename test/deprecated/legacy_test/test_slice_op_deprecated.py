#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from op_test import paddle_static_guard

import paddle

paddle.enable_static()


class TestInferShape(unittest.TestCase):
    def test(self):
        with paddle_static_guard():
            x = paddle.ones(shape=[3, 4, 5])
            x.desc.set_shape([3, -1, 5])
            self.assertEqual(x.shape, (3, -1, 5))

            out0 = paddle.slice(x, axes=[1], starts=[0], ends=[3])
            self.assertEqual(out0.shape, (3, -1, 5))


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

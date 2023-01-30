# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import unittest

import paddle
from paddle.fluid.layers.utils import try_set_static_shape_tensor


class StaticShapeInferrenceTest(unittest.TestCase):
    def test_static_graph(self):
        paddle.enable_static()
        data = paddle.static.data(name="x", shape=[-1, 2], dtype='float32')
        shape = paddle.shape(data)  # shape should be [-1, 2]
        x = paddle.uniform(shape)
        try_set_static_shape_tensor(x, shape)
=======
import paddle
import unittest


class StaticShapeInferrenceTest(unittest.TestCase):

    def test_static_graph(self):
        paddle.enable_static()
        data = paddle.fluid.layers.data(name="x",
                                        shape=[-1, 2],
                                        dtype='float32')
        shape = paddle.fluid.layers.shape(data)  # shape should be [-1, 2]
        x = paddle.fluid.layers.uniform_random(shape)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertEqual(x.shape, data.shape)
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()

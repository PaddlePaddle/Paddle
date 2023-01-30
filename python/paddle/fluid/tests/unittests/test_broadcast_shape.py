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

import unittest
<<<<<<< HEAD

=======
import numpy as np
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle


class TestBroadcastShape(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_result(self):
        shape = paddle.broadcast_shape([2, 1, 3], [1, 3, 1])
        self.assertEqual(shape, [2, 3, 3])

        shape = paddle.broadcast_shape(
<<<<<<< HEAD
            [-1, 1, 3], [1, 3, 1]
        )  # support compile time infershape
        self.assertEqual(shape, [-1, 3, 3])

    def test_error(self):
        self.assertRaises(
            ValueError, paddle.broadcast_shape, [2, 1, 3], [3, 3, 1]
        )
=======
            [-1, 1, 3], [1, 3, 1])  #support compile time infershape
        self.assertEqual(shape, [-1, 3, 3])

    def test_error(self):
        self.assertRaises(ValueError, paddle.broadcast_shape, [2, 1, 3],
                          [3, 3, 1])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == "__main__":
    unittest.main()

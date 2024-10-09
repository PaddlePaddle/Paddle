#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import paddle


class DotOpEmptyInput(unittest.TestCase):
    def test_1d_input(self):
        data = np.array([], dtype=np.float32)
        x = paddle.to_tensor(np.reshape(data, [0]), dtype='float32')
        y = paddle.to_tensor(np.reshape(data, [0]), dtype='float32')
        np_out = np.dot(data, data)
        pd_out = paddle.dot(x, y)

        self.assertEqual(np_out, pd_out)

    def test_2d_input(self):
        data = np.array([], dtype=np.float32)
        x = paddle.to_tensor(np.reshape(data, [0, 0]), dtype='float32')
        y = paddle.to_tensor(np.reshape(data, [0, 0]), dtype='float32')
        pd_out = paddle.dot(x, y)
        self.assertEqual(
            pd_out.shape,
            [
                0,
            ],
        )

    def test_3d_input_error(self):
        data = np.array([], dtype=np.float32)
        x = paddle.to_tensor(np.reshape(data, [0, 0, 0]), dtype='float32')
        y = paddle.to_tensor(np.reshape(data, [0, 0, 0]), dtype='float32')

        self.assertRaises(Exception, paddle.dot, x, y)


if __name__ == '__main__':
    unittest.main()

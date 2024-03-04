# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle


class TestImperativeNumpyBridge(unittest.TestCase):
    def test_tensor_from_numpy(self):
        data_np = np.array([[2, 3, 1]]).astype('float32')
        paddle.set_device('cpu')
        var2 = paddle.to_tensor(data_np)
        np.testing.assert_array_equal(var2.numpy(), data_np)
        data_np[0][0] = -1
        self.assertEqual(data_np[0][0], -1)
        self.assertNotEqual(var2[0][0].numpy(), -1)

        self.assertFalse(np.array_equal(var2.numpy(), data_np))


if __name__ == '__main__':
    unittest.main()

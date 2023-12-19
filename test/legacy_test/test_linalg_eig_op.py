#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class TestEigAPIError(unittest.TestCase):
    def test_errors(self):
        # The size of input in Eig should not be 0.
        def test_0_size():
            array = np.array([], dtype=np.float32)
            x = paddle.to_tensor(np.reshape(array, [0, 0]), dtype='float32')
            paddle.linalg.eig(x)

        self.assertRaises(ValueError, test_0_size)


if __name__ == '__main__':
    unittest.main()

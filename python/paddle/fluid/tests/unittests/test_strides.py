# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class TestPyLayer(unittest.TestCase):

    def test_simple_pylayer_multiple_output(self):

        x_np = np.random.random(size=[2, 3, 4]).astype('float32')
        x = paddle.to_tensor(x_np)
        self.assertTrue(np.allclose(x.numpy(), x_np))

        x_transposed1 = paddle.transpose(x, perm=[1, 0, 2])
        x_np_transposed1 = x_np.transpose(1, 0, 2)
        self.assertTrue(np.allclose(x_transposed1.numpy(), x_np_transposed1))
        self.assertTrue(x_transposed1.is_contiguous() == False)
        self.assertTrue(x._is_shared_buffer_with(x_transposed1))

        x_c = x_transposed1.contiguous()
        self.assertTrue(np.allclose(x_c.numpy(), x_np_transposed1))
        self.assertTrue(x_c._is_shared_buffer_with(x_transposed1) == False)

        x_transposed2 = paddle.transpose(x_transposed1, perm=[2, 0, 1])
        x_np_transposed2 = x_np_transposed1.transpose(2, 0, 1)
        self.assertTrue(np.allclose(x_transposed2.numpy(), x_np_transposed2))
        self.assertTrue(x_transposed2.is_contiguous() == False)
        self.assertTrue(x._is_shared_buffer_with(x_transposed2))

        y = x_transposed2 + 2
        y_np = x_np_transposed2 + 2
        self.assertTrue(np.allclose(y.numpy(), y_np))
        self.assertTrue(y.is_contiguous())
        self.assertTrue(x._is_shared_buffer_with(y) == False)


if __name__ == '__main__':
    unittest.main()

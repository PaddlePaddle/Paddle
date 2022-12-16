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

import numpy as np

import paddle
from paddle.fluid.executor import Executor
from paddle.fluid.layers import data, mul, zeros
from paddle.tensor import array_write


class TestExecutor(unittest.TestCase):
    def test_mul(self):
        i = zeros(shape=[1], dtype='int64')
        a = data(name='a', shape=[784], dtype='float32')
        array = array_write(x=a, i=i)

        i = paddle.increment(i)
        b = data(
            name='b', shape=[784, 100], dtype='float32', append_batch_size=False
        )
        array_write(x=b, i=i, array=array)

        i = paddle.increment(i)
        out = mul(x=a, y=b)
        array_write(x=out, i=i, array=array)

        a_np = np.random.random((100, 784)).astype('float32')
        b_np = np.random.random((784, 100)).astype('float32')

        exe = Executor()
        res, res_array = exe.run(
            feed={'a': a_np, 'b': b_np}, fetch_list=[out, array]
        )

        self.assertEqual((100, 100), res.shape)
        np.testing.assert_allclose(res, np.dot(a_np, b_np), rtol=1e-05)
        np.testing.assert_allclose(res_array[0], a_np, rtol=1e-05)
        np.testing.assert_allclose(res_array[1], b_np, rtol=1e-05)
        np.testing.assert_allclose(res_array[2], res, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()

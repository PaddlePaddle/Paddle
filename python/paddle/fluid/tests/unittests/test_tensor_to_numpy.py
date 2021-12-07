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

import paddle.fluid as fluid
import unittest
import numpy as np
import six


class TensorToNumpyTest(unittest.TestCase):
    def setUp(self):
        self.shape = [11, 25, 32, 43]

    def test_main(self):
        dtypes = [
            'float32', 'float64', 'int32', 'int64', 'uint8', 'int8', 'bool'
        ]

        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
            places.append(fluid.CUDAPinnedPlace())

        for p in places:
            for dtype in dtypes:
                np_arr = np.reshape(
                    np.array(six.moves.range(np.prod(self.shape))).astype(
                        dtype), self.shape)

                t = fluid.LoDTensor()
                t.set(np_arr, p)

                ret_np_arr = np.array(t)
                self.assertEqual(np_arr.shape, ret_np_arr.shape)
                self.assertEqual(np_arr.dtype, ret_np_arr.dtype)

                all_equal = np.all(np_arr == ret_np_arr)
                self.assertTrue(all_equal)


if __name__ == '__main__':
    unittest.main()

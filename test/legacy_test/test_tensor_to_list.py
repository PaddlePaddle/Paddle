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

import os
import unittest

import numpy as np

import paddle
from paddle import base


class TensorToListTest(unittest.TestCase):
    def setUp(self):
        self.shape = [11, 25, 32, 43]

    def test_tensor_tolist(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
            places.append(base.CUDAPinnedPlace())

        for p in places:
            np_arr = np.reshape(
                np.array(range(np.prod(self.shape))), self.shape
            )
            expectlist = np_arr.tolist()

            t = paddle.to_tensor(np_arr, place=p)
            tensorlist = t.tolist()

            self.assertEqual(tensorlist, expectlist)


if __name__ == '__main__':
    unittest.main()

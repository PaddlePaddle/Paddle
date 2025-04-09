# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base.core import DenseTensor as Tensor


class TestTensorCopyFrom(unittest.TestCase):
    def test_main(self):
        place = paddle.CPUPlace()
        np_value = np.random.random(size=[10, 30]).astype('float32')
        t_src = Tensor()
        t_src.set(np_value, place)
        np.testing.assert_array_equal(np_value, t_src)

        t_dst1 = Tensor()
        t_dst1._copy_from(t_src, place)
        np.testing.assert_array_equal(np_value, t_dst1)

        t_dst2 = Tensor()
        t_dst2._copy_from(t_src, place, 5)
        np.testing.assert_array_equal(np.array(np_value[0:5]), t_dst2)


if __name__ == "__main__":
    unittest.main()

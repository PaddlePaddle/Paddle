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
import numpy as np
import unittest


class TestMain(unittest.TestCase):
    def test_main(self):
        places = [fluid.CPUPlace()]
        if fluid.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
            places.append(fluid.CUDAPinnedPlace())

        value = float(np.random.random(size=[1])[0])

        for p in places:
            np_arr = np.array(value)
            t = fluid.LoDTensor()
            t.set(np_arr, p)
            self.assertTrue(t.shape() == [1])
            self.assertTrue(np.array(t)[0] == value)


if __name__ == '__main__':
    unittest.main()

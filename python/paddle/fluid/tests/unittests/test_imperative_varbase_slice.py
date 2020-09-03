#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid


class TestImperativeVarBaseGetItem(unittest.TestCase):
    def test_getitem_with_long(self):
        with fluid.dygraph.guard():
            data = np.random.random((2, 80, 16128)).astype('float32')
            var = fluid.dygraph.to_variable(data)
            sliced = var[:, 10:, :var.shape[1]]  # var.shape[1] is 80L here
            self.assertEqual(sliced.shape, [2, 70, 80])

            sliced = var[:, var.shape[0]:, var.shape[0]:var.shape[1]]
            self.assertEqual(sliced.shape, [2, 78, 78])

    def test_getitem_with_float(self):
        def test_float_in_slice_item():
            with fluid.dygraph.guard():
                data = np.random.random((2, 80, 16128)).astype('float32')
                var = fluid.dygraph.to_variable(data)
                sliced = var[:, 1.1:, :var.shape[1]]

        self.assertRaises(Exception, test_float_in_slice_item)

        def test_float_in_index():
            with fluid.dygraph.guard():
                data = np.random.random((2, 80, 16128)).astype('float32')
                var = fluid.dygraph.to_variable(data)
                sliced = var[1.1]

        self.assertRaises(Exception, test_float_in_index)


if __name__ == '__main__':
    unittest.main()

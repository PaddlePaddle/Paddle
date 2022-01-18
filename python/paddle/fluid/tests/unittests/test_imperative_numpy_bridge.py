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
import paddle.fluid as fluid
import warnings
from paddle.fluid.framework import _test_eager_guard, _in_eager_mode


class TestImperativeNumpyBridge(unittest.TestCase):
    def func_tensor_from_numpy(self):
        data_np = np.array([[2, 3, 1]]).astype('float32')
        with fluid.dygraph.guard(fluid.CPUPlace()):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                var = fluid.dygraph.to_variable(data_np, zero_copy=True)
                assert "Currently, zero_copy is not supported, and it will be discarded." in str(
                    w[-1].message)
            # Temporally diable zero_copy
            # var = fluid.dygraph.to_variable(data_np, zero_copy=True)
            # self.assertTrue(np.array_equal(var.numpy(), data_np))
            # data_np[0][0] = 4
            # self.assertEqual(data_np[0][0], 4)
            # self.assertEqual(var[0][0].numpy()[0], 4)
            # self.assertTrue(np.array_equal(var.numpy(), data_np))

            var2 = fluid.dygraph.to_variable(data_np, zero_copy=False)
            self.assertTrue(np.array_equal(var2.numpy(), data_np))
            data_np[0][0] = -1
            self.assertEqual(data_np[0][0], -1)
            if _in_eager_mode():
                # eager_mode, var2 is EagerTensor, is not subscriptable
                # TODO(wuweilong): to support slice in eager mode later
                self.assertNotEqual(var2.numpy()[0][0], -1)
            else:
                self.assertNotEqual(var2[0][0].numpy()[0], -1)
            self.assertFalse(np.array_equal(var2.numpy(), data_np))

    def test_func_tensor_from_numpy(self):
        with _test_eager_guard():
            self.func_tensor_from_numpy()
        self.func_tensor_from_numpy()


if __name__ == '__main__':
    unittest.main()

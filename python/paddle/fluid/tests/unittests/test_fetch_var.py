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
import paddle.fluid as fluid
import paddle.fluid.layers as layers


class TestFetchVar(unittest.TestCase):
    def set_input(self):
        self.val = np.array([1, 3, 5]).astype(np.int32)

    def test_fetch_var(self):
        self.set_input()
        x = paddle.tensor.create_tensor(
            dtype="int32", persistable=True, name="x"
        )
        layers.assign(input=self.val, output=x)
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_main_program(), feed={}, fetch_list=[])
        fetched_x = fluid.executor._fetch_var("x")
        np.testing.assert_array_equal(fetched_x, self.val)
        self.assertEqual(fetched_x.dtype, self.val.dtype)


class TestFetchNullVar(TestFetchVar):
    def set_input(self):
        self.val = np.array([]).astype(np.int32)


if __name__ == '__main__':
    unittest.main()

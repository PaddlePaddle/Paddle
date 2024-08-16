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
from paddle import base


class TestFetchVar(unittest.TestCase):
    def set_input(self):
        self.val = np.array([1, 3, 5]).astype(np.int32)
        self.name = "x"

    def test_fetch_var(self):
        self.set_input()
        x = paddle.static.data(
            name=self.name, shape=self.val.shape, dtype="int32"
        )
        x.persistable = True
        exe = base.Executor(base.CPUPlace())
        exe.run(
            base.default_main_program(),
            feed={self.name: self.val},
            fetch_list=[],
        )
        fetched_x = base.executor._fetch_var(x.name)
        np.testing.assert_array_equal(fetched_x, self.val)
        self.assertEqual(fetched_x.dtype, self.val.dtype)


class TestFetchNullVar(TestFetchVar):
    def set_input(self):
        self.val = np.array([]).astype(np.int32)
        self.name = "y"


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

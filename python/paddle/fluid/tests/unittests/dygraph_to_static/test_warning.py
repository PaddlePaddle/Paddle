#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import warnings
from paddle.fluid.dygraph.dygraph_to_static.program_translator import convert_to_static


@paddle.jit.to_static
def fun1():
    a = paddle.to_tensor(1)
    b = paddle.to_tensor(2)
    if a > b:
        b = paddle.to_tensor(3)
    else:
        b = None


class TestReturnNoneInIfelse(unittest.TestCase):

    def test_diff_var(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fun1()
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(
                "Set var to 'None' in ifelse block might lead to error." in str(
                    w[0].message))


if __name__ == '__main__':
    unittest.main()

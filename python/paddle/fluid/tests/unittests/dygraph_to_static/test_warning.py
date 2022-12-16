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
import warnings

import paddle
from paddle.static.nn import cond


@paddle.jit.to_static
def fun1():
    a = paddle.to_tensor(1)
    b = paddle.to_tensor(2)
    if a > b:
        b = paddle.to_tensor(3)
    else:
        b = None


def true_fn():
    return [paddle.to_tensor(1), [paddle.to_tensor(2), paddle.to_tensor(3)]]


def false_fn():
    return [paddle.to_tensor(3), [None, paddle.to_tensor(4)]]


class TestReturnNoneInIfelse(unittest.TestCase):
    def test_dy2static_warning(self):
        paddle.disable_static()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fun1()
            flag = False
            for warn in w:
                if (
                    issubclass(warn.category, UserWarning)
                ) and "Set var to 'None' in ifelse block might lead to error." in str(
                    warn.message
                ):
                    flag = True
                    break
            self.assertTrue(flag)

    def test_cond_warning(self):
        paddle.enable_static()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a = paddle.to_tensor(1)
            b = paddle.to_tensor(2)
            cond(a < b, true_fn, false_fn, return_names=['ret1', 'ret2'])
            flag = False
            for warn in w:
                if (
                    issubclass(warn.category, UserWarning)
                ) and "Set var to 'None' in ifelse block might lead to error." in str(
                    warn.message
                ):
                    flag = True
                    break
            self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.base import core


class EagerStringTensorTestCase(unittest.TestCase):
    def setUp(self):
        self.str_arr = np.array(
            [
                [
                    "15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错"
                ],  # From ChnSentiCorp
                ["One of the very best Three Stooges shorts ever."],
            ]
        )  # From IMDB

    def test_constructor_with_args(self):
        ST1 = core.eager.StringTensor()  # constructor 1
        self.assertEqual(ST1.name, "generated_string_tensor_0")
        self.assertEqual(ST1.shape, [])
        self.assertEqual(ST1.numpy(), '')

        shape = [2, 3]
        ST2 = core.eager.StringTensor(shape, "ST2")  # constructor 2
        self.assertEqual(ST2.name, "ST2")
        self.assertEqual(ST2.shape, shape)
        np.testing.assert_array_equal(
            ST2.numpy(), np.empty(shape, dtype=np.str_)
        )

        ST3 = core.eager.StringTensor(self.str_arr, "ST3")  # constructor 3
        self.assertEqual(ST3.name, "ST3")
        self.assertEqual(ST3.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST3.numpy(), self.str_arr)

        ST4 = core.eager.StringTensor(self.str_arr)  # constructor 4
        self.assertEqual(ST4.name, "generated_string_tensor_1")
        self.assertEqual(ST4.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST4.numpy(), self.str_arr)

        ST5 = core.eager.StringTensor(ST4)  # constructor 5
        self.assertEqual(ST5.name, "generated_string_tensor_2")
        self.assertEqual(ST5.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST5.numpy(), self.str_arr)

        ST6 = core.eager.StringTensor(ST5, "ST6")  # constructor 6
        self.assertEqual(ST6.name, "ST6")
        self.assertEqual(ST6.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST6.numpy(), self.str_arr)

        for st in [ST1, ST2, ST3, ST4, ST5, ST6]:
            # All StringTensors are on cpu place so far.
            self.assertTrue(st.place._equals(core.CPUPlace()))

    def test_constructor_with_kwargs(self):
        shape = [2, 3]
        ST1 = core.eager.StringTensor(dims=shape, name="ST1")  # constructor 2
        self.assertEqual(ST1.name, "ST1")
        self.assertEqual(ST1.shape, shape)
        np.testing.assert_array_equal(
            ST1.numpy(), np.empty(shape, dtype=np.str_)
        )

        ST2 = core.eager.StringTensor(self.str_arr, name="ST2")  # constructor 3
        self.assertEqual(ST2.name, "ST2")
        self.assertEqual(ST2.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST2.numpy(), self.str_arr)

        ST3 = core.eager.StringTensor(ST2, name="ST3")  # constructor 6
        self.assertEqual(ST3.name, "ST3")
        self.assertEqual(ST3.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST3.numpy(), self.str_arr)

        ST4 = core.eager.StringTensor(value=ST2, name="ST4")  # constructor 6
        self.assertEqual(ST4.name, "ST4")
        self.assertEqual(ST4.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST4.numpy(), self.str_arr)
        for st in [ST1, ST2, ST3, ST4]:
            # All StringTensors are on cpu place so far.
            self.assertTrue(st.place._equals(core.CPUPlace()))


if __name__ == "__main__":
    unittest.main()

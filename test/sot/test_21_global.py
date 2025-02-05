# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit import sot

global_x = 1
global_y = paddle.to_tensor(2)
global_z = None
global_del_val = 1
global_dict = {}
global_list = [1, 2]
global_inline = 0


def global_func_int():
    global global_x
    global_x = global_x + 1
    return global_x


def global_func_int_add():
    global global_x
    global_x = global_x + global_x
    return global_x + global_x


def global_func_tensor_int_add(tensor_y: paddle.Tensor):
    global global_x
    global_x += 1
    return global_x + tensor_y


def global_multiple_update():
    global global_x
    global_x = 999
    global_x = 888
    global_x = 777
    return global_x - 1


def global_func_tensor():
    global global_y
    global_y = global_y + global_y
    return global_y


def global_func_tensor_add():
    global global_y
    global_y = global_y + global_y
    return global_y + global_y


def global_func():
    global global_x
    global global_y
    global global_z

    global_z = global_x + global_y
    return global_z


def global_del_global():
    global global_del_val

    del global_del_val


def global_func_dict():
    global global_dict
    global_dict["key"] = "value"
    global_dict.update({"test_key1": "test_value2"})
    return global_dict


def global_func_control1():
    global global_dict
    if "key" in global_dict:
        del global_dict["key"]
    return global_dict


def global_func_control2():
    global global_list
    for i in range(len(global_list)):
        global_list[i] = global_list[i] + 1
    return global_list


def global_func_inline_inner_1():
    global global_inline
    global_func_inline_inner_2()
    global_inline += 1


def global_func_inline_inner_2():
    global global_inline
    global_inline += 1


def global_func_inline():
    global_func_inline_inner_1()
    global global_inline
    return global_inline


class TestGlobal(TestCaseBase):
    def test_global_func_int(self):
        global global_x
        self.assert_results_with_global_check(global_func_int, ["global_x"])
        global_x += 1
        self.assert_results_with_global_check(global_func_int, ["global_x"])
        self.assert_results_with_global_check(global_func_int_add, ["global_x"])

    def test_global_multiple_update(self):
        self.assert_results_with_global_check(
            global_multiple_update, ["global_x"]
        )

    def test_global_func_tensor_int_add(self):
        self.assert_results_with_global_check(
            global_func_tensor_int_add, ["global_x"], paddle.to_tensor(1)
        )

    def test_global_func_tensor(self):
        self.assert_results_with_global_check(global_func_tensor, ["global_y"])
        self.assert_results_with_global_check(
            global_func_tensor_add, ["global_y"]
        )

    def test_global_func(self):
        self.assert_results_with_global_check(global_func, ["global_z"])
        self.assertIn("global_del_val", global_del_global.__globals__)
        sot.symbolic_translate(global_del_global)()
        self.assertNotIn("global_del_val", global_del_global.__globals__)

    def test_global_func_dict(self):
        self.assert_results_with_global_check(global_func_dict, ["global_dict"])
        self.assert_results_with_global_check(
            global_func_control1, ["global_dict"]
        )

    def test_global_func_list(self):
        self.assert_results_with_global_check(
            global_func_control2, ["global_list"]
        )

    def test_global_func_inline(self):
        global global_inline
        global_inline = 0
        sot.symbolic_translate(global_func_inline)()
        self.assertEqual(global_inline, 2)
        sot.symbolic_translate(global_func_inline)()
        self.assertEqual(global_inline, 4)


if __name__ == "__main__":
    unittest.main()

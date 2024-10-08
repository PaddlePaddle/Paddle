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

import unittest

from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
    test_sot_only,
)

import paddle


def func_loop_write_dict(x):
    res = {"a": 1}
    t = paddle.shape(x)[0]
    for i in range(t):
        res["a"] = i
    return res


def func_loop_write_list(x):
    res = [1]
    t = paddle.shape(x)[0]
    for i in range(t):
        res[0] = i
    return res


def func_loop_write_nest_dict_list(x):
    res = {"a": [1]}
    t = paddle.shape(x)[0]
    for i in range(t):
        res["a"][0] = i
    return res


def func_loop_write_nest_list_dict(x):
    res = [{"a": 1}]
    t = paddle.shape(x)[0]
    for i in range(t):
        res[0]["a"] = i
    return res


def func_ifelse_write_dict(x):
    res = {"a": 1}
    t = paddle.shape(x)[0]

    if t > 2:
        res["a"] = 2
    else:
        res["a"] = 3
    return res


def func_ifelse_write_list(x):
    res = [1]
    t = paddle.shape(x)[0]

    if t > 2:
        res[0] = 2
    else:
        res[0] = 3
    return res


def func_ifelse_write_nest_dict_list(x):
    res = {"a": [1]}
    t = paddle.shape(x)[0]

    if t > 2:
        res["a"][0] = 2
    else:
        res["a"][0] = 3
    return res


def func_ifelse_write_nest_list_dict(x):
    res = [{"a": 1}]
    t = paddle.shape(x)[0]

    if t > 2:
        res[0]["a"] = 2
    else:
        res[0]["a"] = 3
    return res


class TestWriteContainer(Dy2StTestBase):
    def setUp(self):
        self.set_func()
        self.set_getitem_path()

    def set_func(self):
        self.func = func_loop_write_dict

    def set_getitem_path(self):
        self.getitem_path = ("a",)

    def get_raw_value(self, container, getitem_path):
        out = container
        for path in getitem_path:
            out = out[path]
        return out

    @test_sot_only
    def test_write_container_sot(self):
        func_static = paddle.jit.to_static(self.func)
        input = paddle.to_tensor([1, 2, 3])
        out_static = self.get_raw_value(func_static(input), self.getitem_path)
        out_dygraph = self.get_raw_value(self.func(input), self.getitem_path)
        self.assertEqual(out_static, out_dygraph)

    @test_ast_only
    def test_write_container(self):
        func_static = paddle.jit.to_static(self.func)
        input = paddle.to_tensor([1, 2, 3])
        out_static = self.get_raw_value(
            func_static(input), self.getitem_path
        ).item()
        out_dygraph = self.get_raw_value(self.func(input), self.getitem_path)
        self.assertEqual(out_static, out_dygraph)


class TestLoopWriteContainerList(TestWriteContainer):
    def set_func(self):
        self.func = func_loop_write_list

    def set_getitem_path(self):
        self.getitem_path = (0,)


class TestLoopWriteContainerNestDictList(TestWriteContainer):
    def set_func(self):
        self.func = func_loop_write_nest_dict_list

    def set_getitem_path(self):
        self.getitem_path = ("a", 0)


class TestLoopWriteContainerNestListDict(TestWriteContainer):
    def set_func(self):
        self.func = func_loop_write_nest_list_dict

    def set_getitem_path(self):
        self.getitem_path = (0, "a")


class TestIfElseWriteContainerDict(TestWriteContainer):
    def set_func(self):
        self.func = func_ifelse_write_dict

    def set_getitem_path(self):
        self.getitem_path = ("a",)


class TestIfElseWriteContainerList(TestWriteContainer):
    def set_func(self):
        self.func = func_ifelse_write_list

    def set_getitem_path(self):
        self.getitem_path = (0,)


class TestIfElseWriteContainerNestDictList(TestWriteContainer):
    def set_func(self):
        self.func = func_ifelse_write_nest_dict_list

    def set_getitem_path(self):
        self.getitem_path = ("a", 0)


class TestIfElseWriteContainerNestListDict(TestWriteContainer):
    def set_func(self):
        self.func = func_ifelse_write_nest_list_dict

    def set_getitem_path(self):
        self.getitem_path = (0, "a")


if __name__ == '__main__':
    unittest.main()

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

import paddle


def func_dict(x):
    res = {"a": 1}
    t = paddle.shape(x)[0]
    for i in range(t):
        res["a"] = i
    return res


def func_list(x):
    res = [1]
    t = paddle.shape(x)[0]
    for i in range(t):
        res[0] = i
    return res


def func_nest_dict_list(x):
    res = {"a": [1]}
    t = paddle.shape(x)[0]
    for i in range(t):
        res["a"][0] = i
    return res


def func_nest_list_dict(x):
    res = [{"a": 1}]
    t = paddle.shape(x)[0]
    for i in range(t):
        res[0]["a"] = i
    return res


class TestWriteContainer(unittest.TestCase):
    def setUp(self):
        self.set_func()

    def set_func(self):
        self.func = func_dict

    def test_write_container(self):
        func_static = paddle.jit.to_static(self.func)
        x = paddle.to_tensor([1, 2, 3])
        func_static(x)


class TestWriteContainerList(TestWriteContainer):
    def set_func(self):
        self.func = func_list


class TestWriteContainerNestDictList(TestWriteContainer):
    def set_func(self):
        self.func = func_nest_dict_list


class TestWriteContainerNestListDict(TestWriteContainer):
    def set_func(self):
        self.func = func_nest_list_dict


if __name__ == '__main__':
    unittest.main()

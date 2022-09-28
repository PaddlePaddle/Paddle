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
import paddle
from paddle.fluid.dygraph.dygraph_to_static.utils import GetterSetterHelper

vars = [1, 2, 3, 4, 5]


def getter():
    return vars


def setter(values):
    global vars
    vars = values


class TestGetterSetterHelper(unittest.TestCase):

    def test_1(self):
        helper = GetterSetterHelper(getter, setter, ['a', 'b', 'e'],
                                    ['d', 'f', 'e'])
        print(helper.union())
        expect_union = ['a', 'b', 'd', 'e', 'f']
        assert helper.union() == expect_union
        assert helper.get(expect_union) == (1, 2, 3, 4, 5)
        helper.set(['a', 'b'], [1, 1])
        assert vars == [1, 1, 3, 4, 5]
        helper.set(['f', 'e'], [12, 10])
        assert vars == [1, 1, 3, 10, 12]
        helper.set(None, None)
        assert vars == [1, 1, 3, 10, 12]
        assert helper.get(None) == tuple()
        assert helper.get([]) == tuple()


if __name__ == '__main__':
    unittest.main()

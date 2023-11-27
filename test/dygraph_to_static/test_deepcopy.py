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
from copy import deepcopy

import numpy as np
from dygraph_to_static_utils import Dy2StTestBase
from test_rollback import Net, foo

import paddle
from paddle.jit.dy2static.program_translator import StaticFunction


class TestDeepCopy(Dy2StTestBase):
    def test_net(self):
        net = Net()
        net = paddle.jit.to_static(net)

        x = paddle.randn([3, 4])
        src_out = net(x)
        self.assertTrue(isinstance(net.forward, StaticFunction))

        copy_net = deepcopy(net)
        copy_out = copy_net(x)

        self.assertFalse(isinstance(net.forward, StaticFunction))
        self.assertTrue(id(copy_net), id(copy_net.forward.__self__))
        np.testing.assert_array_equal(src_out.numpy(), copy_out.numpy())

    def test_func(self):
        st_foo = paddle.jit.to_static(foo)
        x = paddle.randn([3, 4])
        st_out = st_foo(x)

        self.assertTrue(isinstance(st_foo, StaticFunction))

        new_foo = deepcopy(st_foo)
        self.assertFalse(isinstance(new_foo, StaticFunction))
        new_out = new_foo(x)
        np.testing.assert_array_equal(st_out.numpy(), new_out.numpy())


if __name__ == "__main__":
    unittest.main()

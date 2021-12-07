#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

from paddle.utils import gast
import unittest

import numpy as np
import paddle.fluid as fluid

from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import create_fill_constant_node
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import data_layer_not_check


class TestDataLayerNotCheck(unittest.TestCase):
    def test_create_none_shape(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            d = data_layer_not_check(name="d", shape=(None, -1, 3))
            self.assertEqual(d.shape, (-1, -1, 3))
            self.assertEqual(d.name, "d")

    def test_feed_mismatch_shape(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            d = data_layer_not_check(name="d", shape=(1, 2, 3))
        feed_in_data = np.random.uniform(size=[1, 2, 4]).astype(np.float32)
        place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        ret = exe.run(main_program,
                      feed={d.name: feed_in_data},
                      fetch_list=[d.name])
        self.assertTrue(np.allclose(ret, feed_in_data))


class TestVariableTransFunc(unittest.TestCase):
    def test_create_fill_constant_node(self):
        node = create_fill_constant_node("a", 1.0)
        source = "a = paddle.fluid.layers.fill_constant(shape=[1], dtype='float64', value=1.0, name='a')"
        self.assertEqual(
            ast_to_source_code(node).replace('\n', '').replace(' ', ''),
            source.replace(' ', ''))

        node = create_fill_constant_node("b", True)
        source = "b = paddle.fluid.layers.fill_constant(shape=[1], dtype='bool', value=True, name='b')"
        self.assertEqual(
            ast_to_source_code(node).replace('\n', '').replace(' ', ''),
            source.replace(' ', ''))

        node = create_fill_constant_node("c", 4293)
        source = "c = paddle.fluid.layers.fill_constant(shape=[1], dtype='int64', value=4293, name='c')"
        self.assertEqual(
            ast_to_source_code(node).replace('\n', '').replace(' ', ''),
            source.replace(' ', ''))

        self.assertIsNone(create_fill_constant_node("e", None))
        self.assertIsNone(create_fill_constant_node("e", []))


if __name__ == '__main__':
    unittest.main()

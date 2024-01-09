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


class TestDispensable(unittest.TestCase):
    def setUp(self):
        paddle.base.core._set_prim_all_enabled(True)

    def tearDown(self):
        paddle.base.core._set_prim_all_enabled(False)

    def test_dispensable(self):
        def f(x):
            return paddle.split(x, num_or_sections=2)

        f = paddle.jit.to_static(full_graph=True)(f)
        x = paddle.rand((8,))
        x.stop_gradient = False

        op = f.get_concrete_program(x)[1].backward_program.block(0).ops[-1]
        self.assertEqual(
            op.attr('op_role'),
            int(paddle.base.core.op_proto_and_checker_maker.OpRole.Backward),
        )
        self.assertIn('AxisTensor', op.input_names)


if __name__ == '__main__':
    unittest.main()

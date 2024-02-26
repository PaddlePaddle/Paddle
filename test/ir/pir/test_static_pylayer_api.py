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
from paddle.base.libpaddle.pir import (
    build_pipe_for_block,
    get_used_external_value,
)

paddle.enable_static()


class TestConstructModuleWithPyLayerOp(unittest.TestCase):
    def test_fwd_only_with_single_output(self):
        """
        pseudocode:

        y = 3 * x
        """

        def forward_fn(x):
            return 3 * x

        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
                x.stop_gradient = True
                out = paddle.static.nn.static_pylayer(
                    forward_fn, [x], backward_fn=None
                )
                y = paddle.mean(out)

            pylayer_op = main_program.global_block().ops[-2]
            self.assertEqual(pylayer_op.name(), "pd_op.pylayer")
            self.assertEqual(len(pylayer_op.results()), 1)
            value_list = get_used_external_value(pylayer_op)
            self.assertEqual(len(value_list), 2)
            self.assertTrue(value_list[0].is_same(pylayer_op.operand_source(0)))
            self.assertTrue(
                value_list[1].is_same(
                    pylayer_op.operand_source(0)
                    .get_defining_op()
                    .operand_source(0)
                )
            )

    def test_fwd_only_with_multi_inputs_multi_outpus(self):
        """
        pseudocode:

        ret1 = x * y
        ret2 = x - y
        """

        def forward_fn(x, y):
            z = 3 * x
            return z * y, z - y

        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
                y = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
                x.stop_gradient = True
                y.stop_gradient = True
                ret1, ret2 = paddle.static.nn.static_pylayer(
                    forward_fn, [x, y], backward_fn=None
                )
                out = ret1 + ret2

            pylayer_op = main_program.global_block().ops[-2]
            self.assertEqual(pylayer_op.name(), "pd_op.pylayer")
            self.assertEqual(len(pylayer_op.results()), 2)
            value_list = get_used_external_value(pylayer_op)
            self.assertEqual(len(value_list), 3)
            self.assertTrue(value_list[0].is_same(pylayer_op.operand_source(0)))
            self.assertTrue(
                value_list[1].is_same(
                    pylayer_op.operand_source(0)
                    .get_defining_op()
                    .operand_source(0)
                )
            )
            self.assertTrue(
                value_list[2].is_same(
                    pylayer_op.operand_source(0)
                    .get_defining_op()
                    .operand_source(1)
                )
            )

            # check build_pipe_for_block interface
            fwd_block = pylayer_op.as_pylayer_op().forward_block()
            build_pipe_for_block(fwd_block)

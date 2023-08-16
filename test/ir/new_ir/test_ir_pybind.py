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
from paddle import ir

paddle.enable_static()


def get_ir_program():
    x = paddle.randn([4, 4])
    main_program, start_program = (
        paddle.static.Program(),
        paddle.static.Program(),
    )
    with paddle.static.program_guard(main_program, start_program):
        x_s = paddle.static.data('x', [4, 4], x.dtype)
        x_s.stop_gradient = False
        y_s = paddle.matmul(x_s, x_s)
        z_s = paddle.add(y_s, y_s)
        k_s = paddle.tanh(z_s)
    newir_program = ir.translate_to_new_ir(main_program.desc)
    return newir_program


class TestPybind(unittest.TestCase):
    def test_program(self):
        newir_program = get_ir_program()
        print(newir_program)

        block = newir_program.block()
        program = block.get_parent_program()

        self.assertEqual(newir_program, program)

    def test_block(self):
        newir_program = get_ir_program()
        block = newir_program.block()
        ops = block.ops
        self.assertEqual(
            len(ops), 4
        )  # ir program add "builtin.get_parameter" by default, so size is 4
        block.remove_op(ops[3])
        self.assertEqual(len(block.ops), 3)

    def test_operation(self):
        newir_program = get_ir_program()
        ops = newir_program.block().ops
        matmul_op = newir_program.block().ops[1]
        add_op = newir_program.block().ops[2]
        tanh_op = newir_program.block().ops[3]
        parent_block = tanh_op.get_parent_block()
        parent_ops_num = len(parent_block.ops)
        self.assertEqual(parent_ops_num, 4)
        self.assertEqual(tanh_op.num_results(), 1)
        self.assertEqual(len(matmul_op.get_input_names()), 2)
        self.assertEqual(len(matmul_op.get_attr_names()), 2)
        self.assertEqual(len(matmul_op.get_output_names()), 1)

    def test_value(self):
        newir_program = get_ir_program()
        matmul_op = newir_program.block().ops[1]
        add_op = newir_program.block().ops[2]
        tanh_op = newir_program.block().ops[3]

        self.assertEqual(
            matmul_op.result(0).dtype, paddle.base.core.DataType.FLOAT32
        )
        self.assertEqual(matmul_op.result(0).shape, [4, 4])
        self.assertEqual(
            matmul_op.results()[0].get_defining_op().name(), "pd.matmul"
        )
        self.assertEqual(
            matmul_op.result(0).get_defining_op().name(), "pd.matmul"
        )
        matmul_op.result(0).stop_gradient = True
        self.assertEqual(matmul_op.result(0).stop_gradient, True)

        # test opresult hash
        result_set = set()
        for opresult in matmul_op.results():
            result_set.add(opresult)
        # test opresult hash and hash(opresult) == hash(operesult)
        self.assertTrue(add_op.operands()[0].source() in result_set)
        # test value hash and hash(value) == hash(operesult)
        self.assertTrue(add_op.operands_source()[0] in result_set)
        # test value == value
        self.assertEqual(
            add_op.operands_source()[0], add_op.operands_source()[0]
        )
        # test value == opresult
        self.assertEqual(add_op.operands_source()[0], matmul_op.results()[0])
        # test opresult == value
        self.assertEqual(
            add_op.operands()[0].source(), add_op.operands_source()[0]
        )
        # test opresult == opresult
        self.assertEqual(add_op.operands()[0].source(), matmul_op.results()[0])

        self.assertEqual(
            tanh_op.operands()[0].source().get_defining_op().name(), "pd.add"
        )

        add_op.replace_all_uses_with(matmul_op.results())
        self.assertEqual(
            tanh_op.operands()[0].source().get_defining_op().name(), "pd.matmul"
        )

        self.assertEqual(add_op.result(0).use_empty(), True)

    def test_type(self):
        newir_program = get_ir_program()
        matmul_op = newir_program.block().ops[1]
        add_op = newir_program.block().ops[2]
        print(matmul_op.result(0).type())
        self.assertEqual(
            matmul_op.result(0).type() == add_op.result(0).type(), True
        )

    def test_attr(self):
        main_program, start_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        with paddle.static.program_guard(main_program, start_program):
            conv_data = paddle.static.data(
                'conv_data', [None, 3, 32, 32], dtype='float32'
            )
            conv2d_out = paddle.static.nn.conv2d(
                input=conv_data,
                num_filters=2,
                filter_size=3,
                stride=3,
                act="relu",
            )
            full_out = paddle.tensor.fill_constant(
                shape=[4, 4], dtype="float32", value=2
            )

        newir_program = ir.translate_to_new_ir(main_program.desc)
        print(newir_program)
        conv_attr = newir_program.block().ops[3].attrs()
        full_attr = newir_program.block().ops[8].attrs()
        self.assertEqual(conv_attr["stop_gradient"], [False])
        self.assertEqual(conv_attr["dilations"], [1, 1])
        self.assertEqual(conv_attr["data_format"], "NCHW")
        self.assertEqual(conv_attr["strides"], [3, 3])
        self.assertEqual(conv_attr["paddings"], [0, 0])
        self.assertEqual(conv_attr["padding_algorithm"], "EXPLICIT")
        self.assertEqual(conv_attr["groups"], 1)
        self.assertEqual(full_attr["dtype"], paddle.base.core.DataType.FLOAT32)
        self.assertTrue(isinstance(full_attr["place"], paddle.base.core.Place))

    def test_operands(self):
        newir_program = get_ir_program()
        matmul_op = newir_program.block().ops[1]
        operands = matmul_op.operands()
        self.assertEqual(len(operands), 2)

    def test_results(self):
        newir_program = get_ir_program()
        matmul_op = newir_program.block().ops[1]
        results = matmul_op.results()
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()

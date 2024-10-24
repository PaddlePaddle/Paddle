# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
import tempfile
import unittest

import paddle
from paddle import base
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock


class TestSaveModuleWithCommonOp(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        paddle.enable_static()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_load(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            input = paddle.full(
                shape=[1, 512, 64], fill_value=0.5, dtype='float32'
            )
            weight = paddle.full(
                shape=[64, 64], fill_value=0.5, dtype='float32'
            )
            bias = paddle.full(shape=[64], fill_value=1.0, dtype='float32')
            x = paddle.matmul(input, weight)
            y = paddle.add(x, bias)

        file_path = os.path.join(self.temp_dir.name, "test_save_program1.json")
        pir_version = 1
        base.core.serialize_pir_program(main_program, file_path, pir_version)

        recover_program = paddle.static.Program()
        base.core.deserialize_pir_program(
            file_path, recover_program, pir_version
        )

        self.assertEqual(
            len(main_program.global_block().ops),
            len(recover_program.global_block().ops),
        )
        for i in range(len(main_program.global_block().ops)):
            self.assertEqual(
                main_program.global_block().ops[i].name(),
                recover_program.global_block().ops[i].name(),
            )

    def test_save_no_trainable(self):
        # check save with trainable=False, no stopgradient info
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            input = paddle.full(
                shape=[1, 512, 64], fill_value=0.5, dtype='float32'
            )
            weight = paddle.full(
                shape=[64, 64], fill_value=0.5, dtype='float32'
            )
            input.stop_gradient = False
            bias = paddle.full(shape=[64], fill_value=1.0, dtype='float32')
            x = paddle.matmul(input, weight)
            y = paddle.add(x, bias)

        file_path = os.path.join(
            self.temp_dir.name, "test_save_program1_0.json"
        )
        pir_version = 1
        base.core.serialize_pir_program(
            main_program, file_path, pir_version, True, True, False
        )

        recover_program = paddle.static.Program()
        base.core.deserialize_pir_program(
            file_path, recover_program, pir_version
        )

        self.assertEqual(
            main_program.global_block().ops[-1].result(0).stop_gradient,
            False,
        )
        self.assertEqual(
            recover_program.global_block().ops[-1].result(0).stop_gradient,
            True,
        )

    def test_builtin_save(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x_2 = paddle.static.data(shape=[4, 5], dtype='int32', name='x_2')
            out1, out2 = paddle.split(x=x_2, num_or_sections=2, axis=0)
            out = paddle.concat([out1, out2], axis=1)

        file_path = os.path.join(self.temp_dir.name, "test_save_program2.json")
        pir_version = 1
        base.core.serialize_pir_program(
            main_program, file_path, pir_version, True, True, True
        )

        recover_program = paddle.static.Program()
        base.core.deserialize_pir_program(
            file_path, recover_program, pir_version
        )

        self.assertEqual(
            len(main_program.global_block().ops),
            len(recover_program.global_block().ops),
        )
        for i in range(len(main_program.global_block().ops)):
            self.assertEqual(
                main_program.global_block().ops[i].name(),
                recover_program.global_block().ops[i].name(),
            )


def true_func():
    a = paddle.full(shape=[1, 2], dtype='float32', fill_value=1)
    b = paddle.full(shape=[2, 3], dtype='int64', fill_value=1)
    return a, b


def false_func():
    a = paddle.full(shape=[1, 2], dtype='float32', fill_value=3)
    b = paddle.full(shape=[2, 3], dtype='int64', fill_value=2)
    return a, b


class TestSaveModuleWithIfOp(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        paddle.enable_static()

    def tearDown(self):
        self.temp_dir.cleanup()

    def construct_program_with_if(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
            y = paddle.static.data(name="y", shape=[6, 1], dtype="float32")
            x.stop_gradient = False
            y.stop_gradient = False
            paddle.static.nn.cond(x < y, lambda: x + y, lambda: x - y)
        return main_program

    def check_block(self, org_block, load_block):
        self.assertEqual(len(org_block.ops), len(load_block.ops))
        for i in range(len(org_block.ops)):
            org_op = org_block.ops[i]
            load_op = load_block.ops[i]
            self.assertEqual(
                org_op.name(),
                load_op.name(),
            )

            for org_block_in, load_block_in in zip(
                org_op.blocks(), load_op.blocks()
            ):
                self.check_block(org_block_in, load_block_in)

    def test_if_with_single_output(self):
        main_program = self.construct_program_with_if()
        file_path = os.path.join(
            self.temp_dir.name, "test_save_program_if.json"
        )
        pir_version = 1
        base.core.serialize_pir_program(main_program, file_path, pir_version)

        recover_program = paddle.static.Program()
        base.core.deserialize_pir_program(
            file_path, recover_program, pir_version
        )

        self.check_block(
            main_program.global_block(), recover_program.global_block()
        )

    def test_if_with_multiple_output(self):
        main_program = self.construct_program_with_if()
        cond_value = main_program.global_block().ops[-1].operand_source(0)
        with paddle.pir.core.program_guard(main_program):
            paddle.static.nn.cond(cond_value, true_func, false_func)

        file_path = os.path.join(
            self.temp_dir.name, "test_save_program_if2.json"
        )
        pir_version = 1
        base.core.serialize_pir_program(main_program, file_path, pir_version)

        recover_program = paddle.static.Program()
        base.core.deserialize_pir_program(
            file_path, recover_program, pir_version
        )

        self.check_block(
            main_program.global_block(), recover_program.global_block()
        )


def cond(i, ten):
    return i < ten


def body(i, ten):
    i = i + 1
    (i,) = paddle.static.nn.while_loop(
        lambda p: p < ten, lambda p: [p + 3], [i]
    )
    return [i, ten]


class TestSaveModuleWithwhileOp(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        paddle.enable_static()

    def tearDown(self):
        self.temp_dir.cleanup()

    def check_block(self, org_block, load_block):
        self.assertEqual(len(org_block.ops), len(load_block.ops))
        for i in range(len(org_block.ops)):
            org_op = org_block.ops[i]
            load_op = load_block.ops[i]
            self.assertEqual(
                org_op.name(),
                load_op.name(),
            )

            for org_block_in, load_block_in in zip(
                org_op.blocks(), load_op.blocks()
            ):
                self.check_block(org_block_in, load_block_in)

    def construct_program_with_while(self):
        main_program = paddle.static.Program()
        with paddle.pir.core.program_guard(main_program):
            i = paddle.full(
                shape=[1], fill_value=0, dtype='int64'
            )  # loop counter
            ten = paddle.full(
                shape=[1], fill_value=10, dtype='int64'
            )  # loop length
            i.stop_gradient = False
            i, ten = paddle.static.nn.while_loop(cond, body, [i, ten])
            return main_program

    def test_while_base(self):
        main_program = self.construct_program_with_while()
        file_path = os.path.join(
            self.temp_dir.name, "test_save_program_while.json"
        )
        pir_version = 1
        base.core.serialize_pir_program(main_program, file_path, pir_version)

        recover_program = paddle.static.Program()
        base.core.deserialize_pir_program(
            file_path, recover_program, pir_version
        )

        self.check_block(
            main_program.global_block(), recover_program.global_block()
        )

    def test_get_used_external_value(self):
        main_program = paddle.static.Program()
        with paddle.pir.core.program_guard(main_program):
            i = paddle.full(shape=[1], fill_value=0)
            x = paddle.full(shape=[1], fill_value=10)
            y = paddle.full(shape=[1], fill_value=5)
            # i, x = paddle.static.nn.while_loop(cond, body, [i, ten])
            paddle.static.nn.while_loop(
                lambda p, q: p < q, lambda p, q: [p + y, q + i], [i, x]
            )

        file_path = os.path.join(
            self.temp_dir.name, "test_save_program_while.json"
        )
        pir_version = 1
        base.core.serialize_pir_program(main_program, file_path, pir_version)

        recover_program = paddle.static.Program()
        base.core.deserialize_pir_program(
            file_path, recover_program, pir_version
        )

        self.check_block(
            main_program.global_block(), recover_program.global_block()
        )

    def test_nested_net(self):
        def external_cond(i, j, init, sums):
            return paddle.less_than(i, loop_len1)

        def external_body(i, j, init, sums):
            def internal_cond(j, init, sums):
                return paddle.less_than(j, loop_len2)

            def internal_body(j, init, sums):
                init = paddle.add(x=init, y=ones)
                sums = paddle.add(x=init, y=sums)
                j = paddle.increment(j)
                return [j, init, sums]

            result = paddle.static.nn.while_loop(
                internal_cond, internal_body, [j, init, sums]
            )
            j = result[0]
            init = result[1]
            sums = result[2]
            sums = paddle.add(x=init, y=sums)
            i = paddle.increment(i)
            return [i, j, init, sums]

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.zeros(shape=[1], dtype='int64')
            j = paddle.zeros(shape=[1], dtype='int64')
            init = paddle.static.data(
                name='init', shape=[3, 3], dtype='float32'
            )
            sums = paddle.static.data(
                name='sums', shape=[3, 3], dtype='float32'
            )
            loop_len1 = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=2
            )
            loop_len2 = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=3
            )
            ones = paddle.tensor.fill_constant(
                shape=[3, 3], dtype='float32', value=1
            )

            out = paddle.static.nn.while_loop(
                external_cond, external_body, [i, j, init, sums]
            )

        file_path = os.path.join(
            self.temp_dir.name, "test_save_program_while_nest.json"
        )
        pir_version = 1
        base.core.serialize_pir_program(main_program, file_path, pir_version)

        recover_program = paddle.static.Program()
        base.core.deserialize_pir_program(
            file_path, recover_program, pir_version
        )

        self.check_block(
            main_program.global_block(), recover_program.global_block()
        )


class TestJsonToPdmodel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        paddle.disable_static()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_json_to_pdmodel(self):
        net = ResNet(BottleneckBlock, 50)
        net = paddle.jit.to_static(net, full_graph=True)
        save_json = os.path.join(self.temp_dir.name, 'save1')
        save_model = os.path.join(self.temp_dir.name, 'save2')
        input_spec = [
            paddle.static.InputSpec(shape=[1, 3, 224, 224], dtype='float32')
        ]
        paddle.jit.save(net, save_json, input_spec)

        # load and save to pdmodel
        with paddle.pir_utils.OldIrGuard():
            input_spec = [
                paddle.static.InputSpec(shape=[1, 3, 224, 224], dtype='float32')
            ]
        paddle.jit.json_to_pdmodel(net, input_spec, save_json, save_model)
        self.assertTrue(os.path.exists(save_model + '.pdmodel'))


if __name__ == '__main__':
    unittest.main()

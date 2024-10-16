# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy
from utils import compare_legacy_with_pt

import paddle
from paddle import base
from paddle.base import core
from paddle.base.backward import append_backward
from paddle.base.executor import Executor
from paddle.base.framework import in_pir_mode
from paddle.incubate.layers.nn import shuffle_batch

paddle.enable_static()


class TestWhileOp(unittest.TestCase):
    def simple_net(self):
        d0 = paddle.static.data("d0", shape=[10], dtype='float32')
        d1 = paddle.static.data("d1", shape=[10], dtype='float32')
        d2 = paddle.static.data("d2", shape=[10], dtype='float32')
        d0.persistable = True
        d0.stop_gradient = False
        d1.persistable = True
        d2.persistable = True
        i = paddle.zeros(shape=[1], dtype='int64')
        i.stop_gradient = True
        i.persistable = True
        init = paddle.zeros(shape=[10], dtype='float32')
        mem_array = paddle.tensor.array_write(x=init, i=i)
        data_array = paddle.tensor.array_write(x=d0, i=i)
        mem_array.stop_gradient = False
        data_array.stop_gradient = False
        mem_array.persistable = True
        i = paddle.increment(i)
        paddle.tensor.array_write(d1, i, array=data_array)
        i = paddle.increment(i)
        paddle.tensor.array_write(d2, i, array=data_array)
        i = paddle.zeros(shape=[1], dtype='int64')
        i.stop_gradient = True
        array_len = paddle.tensor.fill_constant(
            shape=[1], dtype='int64', value=1
        )
        array_len.stop_gradient = True
        cond = paddle.less_than(x=i, y=array_len)
        j = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=1)
        j.stop_gradient = True
        array_len2 = paddle.tensor.fill_constant(
            shape=[1], dtype='int64', value=3
        )
        array_len2.stop_gradient = True
        cond2 = paddle.less_than(x=j, y=array_len2)
        while_op = paddle.static.nn.control_flow.While(cond=cond)
        while_op2 = paddle.static.nn.control_flow.While(cond=cond2)
        with while_op.block():
            d = paddle.tensor.array_read(array=data_array, i=i)
            prev = paddle.tensor.array_read(array=mem_array, i=i)
            result = paddle.add_n([d, prev])

            i = paddle.increment(x=i)
            paddle.tensor.array_write(result, i=i, array=mem_array)

            with while_op2.block():
                d2 = paddle.tensor.array_read(array=data_array, i=j)
                prev2 = paddle.tensor.array_read(array=mem_array, i=j)
                result2 = paddle.add_n([d2, prev2])

                paddle.increment(x=j)
                paddle.tensor.array_write(result2, i=j, array=mem_array)
                paddle.assign(paddle.less_than(x=j, y=array_len2), cond2)

            paddle.assign(paddle.less_than(x=i, y=array_len), cond)
        sum_result = paddle.tensor.array_read(array=mem_array, i=j)
        loss = paddle.mean(sum_result)
        return loss, sum_result

    def test_simple_net(self):
        main_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(main_program, startup_program):
            loss, sum_result = self.simple_net()
            append_backward(loss)

            cpu = core.CPUPlace()
            exe = Executor(cpu)
            d = []

            for i in range(3):
                d.append(numpy.random.random(size=[10]).astype('float32'))

            outs = exe.run(
                feed={'d0': d[0], 'd1': d[1], 'd2': d[2]},
                fetch_list=[sum_result],
            )
            self.assertAlmostEqual(numpy.sum(d), numpy.sum(outs[0]), delta=0.01)

    def test_simple_net_forward(self):
        main_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(main_program, startup_program):
            self.simple_net()
            if in_pir_mode():
                binary = main_program
            else:
                binary = base.compiler.CompiledProgram(main_program)
            cpu = core.CPUPlace()
            exe = Executor(cpu)
            d = []

            for i in range(3):
                d.append(numpy.random.random(size=[10]).astype('float32'))

            for _ in range(2):
                exe.run(binary, feed={'d0': d[0], 'd1': d[1], 'd2': d[2]})

    @compare_legacy_with_pt
    def test_exceptions(self):
        i = paddle.zeros(shape=[2], dtype='int64')
        array_len = paddle.tensor.fill_constant(
            shape=[2], dtype='int64', value=1
        )
        cond = paddle.less_than(x=i, y=array_len)
        with self.assertRaises(TypeError):
            paddle.static.nn.control_flow.While(cond=cond)
        cond = paddle.cast(cond, dtype='float64')
        with self.assertRaises(TypeError):
            paddle.static.nn.control_flow.While(cond=cond)


class BadInputTest(unittest.TestCase):
    @compare_legacy_with_pt
    def test_error(self):
        with base.program_guard(base.Program()):

            def test_bad_x():
                x = [1, 2, 3]
                paddle.increment(x)

            self.assertRaises(TypeError, test_bad_x)


class TestIgnoreVarNameInWhile(unittest.TestCase):

    def test_ignore_var(self):
        def cond(i, ten, temp, y):
            return i < ten

        def body_func(i, ten, batch_info, origin_seq):
            print(batch_info)
            batch_info = shuffle_batch(batch_info)
            print(batch_info)
            i = i + 1
            return [i, ten, batch_info, origin_seq]

        x = paddle.static.data(name='x', shape=[-1, 1, 4], dtype='float32')
        y = paddle.static.data(name='y', shape=[-1, 1, 1], dtype='float32')
        if not in_pir_mode():
            x.desc.set_need_check_feed(False)
            y.desc.set_need_check_feed(False)
        temp = paddle.concat([x, y], axis=-1)

        i = paddle.tensor.fill_constant(shape=[1], value=0, dtype='int32')
        num = paddle.tensor.fill_constant(shape=[1], value=5, dtype='int32')

        i, ten, shuffle_temp, y = paddle.static.nn.while_loop(
            cond, body_func, [i, num, temp, y]
        )

        output = shuffle_temp

        exe = base.Executor(base.CPUPlace())
        exe.run(base.default_startup_program())

        input_x = numpy.array(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 10.0]]
        ).astype('float32')
        input_x = input_x.reshape(3, 1, 4)
        input_y = numpy.array([[10.0], [12.0], [33.0]]).astype('float32')
        input_y = input_y.reshape(3, 1, 1)

        (res,) = exe.run(
            base.default_main_program(),
            feed={'x': input_x, 'y': input_y},
            fetch_list=[output],
        )

        self.assertListEqual(list(res.shape), [3, 1, 5])


class TestOutputsMustExistsInputs(unittest.TestCase):
    @compare_legacy_with_pt
    def test_outputs_exists_inputs(self):
        """
        We guarantee that the output tensor must be in the input tensor, so that the output and input can correspond to each other, but the input can be greater than the number of outputs. It's required in paddle2onnx.
        """
        main_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(main_program, startup_program):

            def func(x):
                s = paddle.zeros([])
                i = paddle.ones([])
                max_len = paddle.shape(x)

                def cond(i, s, x):
                    return i < max_len

                def body(i, s, x):
                    iter = x[i]
                    s += iter
                    i += 1
                    return i, s, x

                [i, s, x] = paddle.static.nn.while_loop(cond, body, [i, s, x])
                return s

            paddle.enable_static()
            x = paddle.static.data(shape=[-1], name='x', dtype='float32')
            func(x)

        # NOTE(winter-wang): The while_op in pir mode doesn't need following constraint, so here only check when in non-pir mode.
        if not in_pir_mode():
            for op in main_program.block(0).ops:
                if op.type == "while":
                    for out_name in op.output("Out"):
                        if out_name in op.input("Condition"):
                            continue
                        self.assertTrue(
                            out_name in op.input("X"),
                            f"In while op, the variable in output(`Out`) must exists in inputs(`X`), but the variable with name `{out_name}` not meet the precondition.",
                        )


if __name__ == '__main__':
    unittest.main()

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

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from paddle.fluid.backward import append_backward
from paddle.fluid.executor import Executor

paddle.enable_static()


class TestWhileOp(unittest.TestCase):
    def simple_net(self):
        d0 = layers.data(
            "d0", shape=[10], append_batch_size=False, dtype='float32'
        )
        d1 = layers.data(
            "d1", shape=[10], append_batch_size=False, dtype='float32'
        )
        d2 = layers.data(
            "d2", shape=[10], append_batch_size=False, dtype='float32'
        )
        i = layers.zeros(shape=[1], dtype='int64')
        i.stop_gradient = True
        init = layers.zeros(shape=[10], dtype='float32')
        mem_array = paddle.tensor.array_write(x=init, i=i)
        data_array = paddle.tensor.array_write(x=d0, i=i)
        i = paddle.increment(i)
        paddle.tensor.array_write(d1, i, array=data_array)
        i = paddle.increment(i)
        paddle.tensor.array_write(d2, i, array=data_array)
        i = layers.zeros(shape=[1], dtype='int64')
        i.stop_gradient = True
        array_len = layers.fill_constant(shape=[1], dtype='int64', value=1)
        array_len.stop_gradient = True
        cond = paddle.less_than(x=i, y=array_len)
        j = layers.fill_constant(shape=[1], dtype='int64', value=1)
        j.stop_gradient = True
        array_len2 = layers.fill_constant(shape=[1], dtype='int64', value=3)
        array_len2.stop_gradient = True
        cond2 = paddle.less_than(x=j, y=array_len2)
        while_op = paddle.static.nn.control_flow.While(cond=cond)
        while_op2 = paddle.static.nn.control_flow.While(cond=cond2)
        with while_op.block():
            d = paddle.tensor.array_read(array=data_array, i=i)
            prev = paddle.tensor.array_read(array=mem_array, i=i)
            result = layers.sums(input=[d, prev])

            i = paddle.increment(x=i)
            paddle.tensor.array_write(result, i=i, array=mem_array)
            paddle.assign(paddle.less_than(x=i, y=array_len), cond)

            with while_op2.block():
                d2 = paddle.tensor.array_read(array=data_array, i=j)
                prev2 = paddle.tensor.array_read(array=mem_array, i=j)
                result2 = layers.sums(input=[d2, prev2])

                j = paddle.increment(x=j)
                paddle.tensor.array_write(result2, i=j, array=mem_array)
                paddle.assign(paddle.less_than(x=j, y=array_len2), cond2)
        sum_result = paddle.tensor.array_read(array=mem_array, i=j)
        loss = paddle.mean(sum_result)
        return loss, sum_result

    def test_simple_net(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
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
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            self.simple_net()
            binary = fluid.compiler.CompiledProgram(main_program)

            cpu = core.CPUPlace()
            exe = Executor(cpu)
            d = []

            for i in range(3):
                d.append(numpy.random.random(size=[10]).astype('float32'))

            for _ in range(2):
                exe.run(binary, feed={'d0': d[0], 'd1': d[1], 'd2': d[2]})

    def test_exceptions(self):
        i = layers.zeros(shape=[2], dtype='int64')
        array_len = layers.fill_constant(shape=[2], dtype='int64', value=1)
        cond = paddle.less_than(x=i, y=array_len)
        with self.assertRaises(TypeError):
            paddle.static.nn.control_flow.While(cond=cond)
        cond = layers.cast(cond, dtype='float64')
        with self.assertRaises(TypeError):
            paddle.static.nn.control_flow.While(cond=cond)


class BadInputTest(unittest.TestCase):
    def test_error(self):
        with fluid.program_guard(fluid.Program()):

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
            batch_info = fluid.contrib.layers.shuffle_batch(batch_info)
            print(batch_info)
            i = i + 1
            return [i, ten, batch_info, origin_seq]

        x = fluid.layers.data(name='x', shape=[-1, 1, 4])
        y = fluid.layers.data(name='y', shape=[-1, 1, 1])
        temp = layers.concat(input=[x, y], axis=-1)
        i = layers.fill_constant(shape=[1], value=0, dtype='int32')
        num = layers.fill_constant(shape=[1], value=5, dtype='int32')

        i, ten, shuffle_temp, y = paddle.static.nn.while_loop(
            cond, body_func, [i, num, temp, y]
        )

        output = shuffle_temp

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        input_x = numpy.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
        input_x = input_x.reshape(3, 1, 4)
        input_y = numpy.array([[10], [12], [33]])
        input_y = input_y.reshape(3, 1, 1)

        (res,) = exe.run(
            fluid.default_main_program(),
            feed={'x': input_x, 'y': input_y},
            fetch_list=[output],
        )

        self.assertListEqual(list(res.shape), [3, 1, 5])


class TestOutputsMustExistsInputs(unittest.TestCase):
    def test_outputs_exists_inputs(self):
        """
        We guarantee that the output tensor must be in the input tensor, so that the output and input can correspond to each other, but the input can be greater than the number of outputs. It's required in paddle2onnx.
        """
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):

            def func(x):
                s = paddle.zeros([1])
                i = paddle.ones([1])
                max_len = paddle.shape(x)[0]

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
            x = paddle.static.data(shape=[-1], name='x')
            func(x)
        for op in main_program.block(0).ops:
            if op.type == "while":
                for out_name in op.output("Out"):
                    if out_name in op.input("Condition"):
                        continue
                    self.assertTrue(
                        out_name in op.input("X"),
                        "In while op, the variable in output(`Out`) must exists in inputs(`X`), but the variable with name `{}` not meet the precondition.".format(
                            out_name
                        ),
                    )


if __name__ == '__main__':
    unittest.main()

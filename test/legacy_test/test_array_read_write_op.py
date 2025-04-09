#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle import base
from paddle.base import Program, core, program_guard
from paddle.base.backward import append_backward
from paddle.base.executor import Executor
from paddle.base.framework import default_main_program


def _test_read_write(x):
    i = paddle.zeros(shape=[1], dtype='int64')
    i.stop_gradient = False
    arr = paddle.tensor.array_write(x=x[0], i=i)
    i = paddle.increment(x=i)
    arr = paddle.tensor.array_write(x=x[1], i=i, array=arr)
    i = paddle.increment(x=i)
    arr = paddle.tensor.array_write(x=x[2], i=i, array=arr)

    i = paddle.zeros(shape=[1], dtype='int64')
    i.stop_gradient = False
    a0 = paddle.tensor.array_read(array=arr, i=i)
    i = paddle.increment(x=i)
    a1 = paddle.tensor.array_read(array=arr, i=i)
    i = paddle.increment(x=i)
    a2 = paddle.tensor.array_read(array=arr, i=i)

    mean_a0 = paddle.mean(a0)
    mean_a1 = paddle.mean(a1)
    mean_a2 = paddle.mean(a2)

    a_sum = paddle.add_n([mean_a0, mean_a1, mean_a2])

    mean_x0 = paddle.mean(x[0])
    mean_x1 = paddle.mean(x[1])
    mean_x2 = paddle.mean(x[2])

    x_sum = paddle.add_n([mean_x0, mean_x1, mean_x2])

    return a_sum, x_sum


class TestArrayReadWrite(unittest.TestCase):
    def test_read_write(self):
        paddle.enable_static()
        x = [
            paddle.static.data(name='x0', shape=[-1, 100]),
            paddle.static.data(name='x1', shape=[-1, 100]),
            paddle.static.data(name='x2', shape=[-1, 100]),
        ]
        for each_x in x:
            each_x.stop_gradient = False

        tensor = np.random.random(size=(100, 100)).astype('float32')
        a_sum, x_sum = _test_read_write(x)

        place = core.CPUPlace()
        exe = Executor(place)
        outs = exe.run(
            feed={'x0': tensor, 'x1': tensor, 'x2': tensor},
            fetch_list=[a_sum, x_sum],
            scope=core.Scope(),
        )
        self.assertEqual(outs[0], outs[1])

        total_sum = paddle.add_n([a_sum, x_sum])
        total_sum_scaled = paddle.scale(x=total_sum, scale=1 / 6.0)

        grad_list = append_backward(total_sum_scaled, [x[0], x[1], x[2]])
        if not paddle.framework.in_pir_mode():
            g_vars = list(
                map(
                    default_main_program().global_block().var,
                    [each_x.name + "@GRAD" for each_x in x],
                )
            )
        else:
            g_vars = []
            for each_x in x:
                for p, g in grad_list:
                    if p.is_same(each_x):
                        g_vars.append(g)
                        continue
        g_out = [
            item.sum()
            for item in exe.run(
                feed={'x0': tensor, 'x1': tensor, 'x2': tensor},
                fetch_list=g_vars,
            )
        ]
        g_out_sum = np.array(g_out).sum()

        # since our final gradient is 1 and the neural network are all linear
        # with mean_op.
        # the input gradient should also be 1
        self.assertAlmostEqual(1.0, g_out_sum, delta=0.1)

        with base.dygraph.guard(place):
            tensor1 = paddle.to_tensor(tensor)
            tensor2 = paddle.to_tensor(tensor)
            tensor3 = paddle.to_tensor(tensor)
            x_dygraph = [tensor1, tensor2, tensor3]
            for each_x in x_dygraph:
                each_x.stop_gradient = False
            a_sum_dygraph, x_sum_dygraph = _test_read_write(x_dygraph)
            self.assertEqual(a_sum_dygraph, x_sum_dygraph)

            total_sum_dygraph = paddle.add_n([a_sum_dygraph, x_sum_dygraph])
            total_sum_scaled_dygraph = paddle.scale(
                x=total_sum_dygraph, scale=1 / 6.0
            )
            total_sum_scaled_dygraph.backward()
            g_out_dygraph = [
                item._grad_ivar().numpy().sum() for item in x_dygraph
            ]
            g_out_sum_dygraph = np.array(g_out_dygraph).sum()

            self.assertAlmostEqual(1.0, g_out_sum_dygraph, delta=0.1)


class TestArrayReadWriteOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            x1 = np.random.randn(2, 4).astype('int32')
            x2 = paddle.ones(shape=[1], dtype='int32')
            x3 = np.random.randn(2, 4).astype('int32')

            self.assertRaises(
                TypeError, paddle.tensor.array_read, array=x1, i=x2
            )
            self.assertRaises(
                TypeError, paddle.tensor.array_write, array=x1, i=x2, out=x3
            )


class TestArrayReadWriteApi(unittest.TestCase):
    def test_api(self):
        paddle.disable_static()
        arr = paddle.tensor.create_array(dtype="float32")
        x = paddle.full(shape=[1, 3], fill_value=5, dtype="float32")
        i = paddle.zeros(shape=[1], dtype="int32")

        arr = paddle.tensor.array_write(x, i, array=arr)

        item = paddle.tensor.array_read(arr, i)

        np.testing.assert_allclose(x.numpy(), item.numpy(), rtol=1e-05)
        paddle.enable_static()


class TestPirArrayOp(unittest.TestCase):
    def test_array(self):
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program = paddle.pir.Program()
            with paddle.static.program_guard(main_program):
                x = paddle.full(shape=[1, 3], fill_value=5, dtype="float32")
                y = paddle.full(shape=[1, 3], fill_value=6, dtype="float32")
                array = paddle.tensor.create_array(
                    dtype="float32", initialized_list=[x]
                )
                array = paddle.tensor.array_write(
                    y, paddle.tensor.array_length(array), array=array
                )
                out0 = paddle.tensor.array_read(array, 0)
                out1 = paddle.tensor.array_read(array, 1)

            place = (
                paddle.base.CPUPlace()
                if not paddle.base.core.is_compiled_with_cuda()
                else paddle.base.CUDAPlace(0)
            )
            exe = paddle.base.Executor(place)
            [fetched_out0, fetched_out1] = exe.run(
                main_program, feed={}, fetch_list=[out0, out1]
            )

        np.testing.assert_array_equal(
            fetched_out0, np.ones([1, 3], dtype="float32") * 5
        )
        np.testing.assert_array_equal(
            fetched_out1, np.ones([1, 3], dtype="float32") * 6
        )

    def test_array_backward(self):
        np.random.seed(2013)
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            d0 = paddle.static.data(name='d0', shape=[10], dtype='float32')
            d0.stop_gradient = False
            d0.persistable = True
            i = paddle.zeros(shape=[1], dtype='int64')
            mem_array = paddle.tensor.array_write(x=d0, i=i)
            mem_array.stop_gradient = False
            mem_array.persistable = True
            out = paddle.tensor.array_read(array=mem_array, i=i)
            mean = paddle.mean(out)
            grad_list = append_backward(mean)

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            d = np.random.random(size=[10]).astype('float32')
            exe = base.Executor(place)

            if paddle.framework.in_pir_mode():
                for p, g in grad_list:
                    if p.is_same(d0):
                        dd0 = g
                    if p.is_same(mem_array):
                        dmem_array = g
                dmem0 = paddle.tensor.array_read(
                    dmem_array, paddle.zeros(shape=[1], dtype='int64')
                )
                res = exe.run(
                    main_program,
                    feed={'d0': d},
                    fetch_list=[mean, dd0, dmem0],  # dmem_array
                )
                # pir not support fetch tensorarray
                np.testing.assert_allclose(res[2], [0.0] * 10, rtol=1e-05)
            else:
                res = exe.run(
                    main_program,
                    feed={'d0': d},
                    fetch_list=[mean.name, d0.grad_name, mem_array.grad_name],
                )
                # this ans is wrong array is empty at beginning ,so it no grad.
                np.testing.assert_allclose(res[2], [[0.1] * 10], rtol=1e-05)

            mean = 0.6097253
            x_grad = [0.1] * 10
            np.testing.assert_allclose(res[0], mean, rtol=1e-05)
            np.testing.assert_allclose(res[1], x_grad, rtol=1e-05)

    def test_create_array_like_add_n(self):
        paddle.enable_static()
        np.random.seed(2013)
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                d0 = paddle.static.data(name='d0', shape=[10], dtype='float32')
                d1 = paddle.static.data(name='d1', shape=[10], dtype='float32')
                i = paddle.zeros(shape=[1], dtype='int64')
                mem_array = paddle.tensor.array_write(x=d0, i=i)
                i = paddle.increment(i)
                paddle.tensor.array_write(x=d1, i=i, array=mem_array)
                copy_array = paddle._pir_ops.create_array_like(mem_array, 0.0)
                out = paddle.tensor.array_read(array=copy_array, i=i)

                paddle.tensor.array_write(x=d0, i=i, array=copy_array)
                i = paddle.increment(i, -1)
                paddle.tensor.array_write(x=d1, i=i, array=copy_array)

                add_array = paddle._pir_ops.add_n_array([mem_array, copy_array])
                out_1 = paddle.tensor.array_read(array=add_array, i=i)
                i = paddle.increment(i, 1)
                out_2 = paddle.tensor.array_read(array=add_array, i=i)

                place = (
                    base.CUDAPlace(0)
                    if core.is_compiled_with_cuda()
                    else base.CPUPlace()
                )
                d0 = np.random.random(size=[10]).astype('float32')
                d1 = np.random.random(size=[10]).astype('float32')
                exe = base.Executor(place)
                res = exe.run(
                    main_program,
                    feed={'d0': d0, 'd1': d1},
                    fetch_list=[out, out_1, out_2],
                )
                out = [0.0] * 10
                np.testing.assert_allclose(res[0], out, rtol=1e-05)
                np.testing.assert_allclose(res[1], d0 + d1, rtol=1e-05)
                np.testing.assert_allclose(res[2], d0 + d1, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()

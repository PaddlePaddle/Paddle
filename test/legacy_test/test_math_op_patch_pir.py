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

import inspect
import unittest
import warnings

import numpy as np

import paddle
from paddle import base

paddle.enable_static()
paddle.device.set_device("cpu")


def new_program():
    # TODO(gouzil): Optimize program code
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    place = paddle.CPUPlace()
    exe = base.Executor(place)
    return (
        main_program,
        exe,
        paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ),
    )


class TestMathOpPatchesPir(unittest.TestCase):
    def test_pow(self):
        # Calculate results in dynamic graphs
        paddle.disable_static()
        x_np = np.random.random([10, 1024]).astype('float32')
        y_np = np.random.random([10, 1024]).astype('float32')
        res_np_b = x_np**y_np
        res_np_c = paddle.pow(paddle.to_tensor(x_np), 2)
        res_np_d = x_np.__pow__(2)
        res_np_e = x_np.__rpow__(2)
        paddle.enable_static()
        # Calculate results under pir
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(
                    name='x', shape=[10, 1024], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[10, 1024], dtype='float32'
                )
                b = x**y
                c = x.pow(2)
                d = x.__pow__(2)
                e = x.__rpow__(2)
                # TODO(gouzil): Why not use `paddle.static.default_main_program()`？
                # Because different case do not isolate parameters (This is a known problem)
                (b_np, c_np, d_np, e_np) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[b, c, d, e],
                )
                np.testing.assert_allclose(res_np_b, b_np, rtol=1e-05)
                np.testing.assert_allclose(res_np_c, c_np, rtol=1e-05)
                np.testing.assert_allclose(res_np_d, d_np, rtol=1e-05)
                np.testing.assert_allclose(res_np_e, e_np, rtol=1e-05)

    def test_mod(self):
        paddle.disable_static()
        x_np = np.random.randint(1, 100, size=[10, 1024], dtype=np.int64)
        y_np = np.random.randint(1, 100, size=[10, 1024], dtype=np.int64)
        res_np_b = x_np % y_np
        res_np_c = paddle.mod(paddle.to_tensor(x_np), paddle.to_tensor(y_np))
        res_np_d = x_np.__mod__(y_np)
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(
                    name='x', shape=[10, 1024], dtype='int64'
                )
                y = paddle.static.data(
                    name='y', shape=[10, 1024], dtype='int64'
                )
                b = x % y
                c = x.mod(y)
                d = x.__mod__(y)
                (b_np, c_np, d_np) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[b, c, d],
                )
                np.testing.assert_allclose(res_np_b, b_np, atol=1e-05)
                np.testing.assert_allclose(res_np_c, c_np, atol=1e-05)
                np.testing.assert_allclose(res_np_d, d_np, atol=1e-05)

    def test_matmul(self):
        paddle.disable_static()
        x_np = np.random.uniform(-1, 1, [2, 3]).astype('float32')
        y_np = np.random.uniform(-1, 1, [3, 5]).astype('float32')
        res_np_b = x_np @ y_np  # __matmul__
        res_np_c = paddle.matmul(paddle.to_tensor(x_np), paddle.to_tensor(y_np))
        res_np_d = x_np.__matmul__(y_np)
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')
                y = paddle.static.data(name='y', shape=[3, 5], dtype='float32')
                b = x @ y
                c = x.matmul(y)
                d = x.__matmul__(y)
                (b_np, c_np, d_np) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[b, c, d],
                )
                np.testing.assert_allclose(res_np_b, b_np, atol=1e-05)
                np.testing.assert_allclose(res_np_c, c_np, atol=1e-05)
                np.testing.assert_allclose(res_np_d, d_np, atol=1e-05)

    def test_floordiv(self):
        paddle.disable_static()
        x_np = np.full([10, 1024], 10, np.int64)
        y_np = np.full([10, 1024], 2, np.int64)
        res_np_b = x_np // y_np
        res_np_c = paddle.floor_divide(
            paddle.to_tensor(x_np), paddle.to_tensor(y_np)
        )
        res_np_d = x_np.__floordiv__(y_np)
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(
                    name='x', shape=[10, 1024], dtype='int64'
                )
                y = paddle.static.data(
                    name='y', shape=[10, 1024], dtype='int64'
                )
                b = x // y
                c = x.floor_divide(y)
                d = x.__floordiv__(y)
                (b_np, c_np, d_np) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[b, c, d],
                )
                np.testing.assert_allclose(res_np_b, b_np, atol=1e-05)
                np.testing.assert_allclose(res_np_c, c_np, atol=1e-05)
                np.testing.assert_allclose(res_np_d, d_np, atol=1e-05)

    def test_bitwise_not(self):
        paddle.disable_static()
        x_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        res_np_b = ~x_np
        res_np_c = paddle.bitwise_not(paddle.to_tensor(x_np))
        res_np_d = x_np.__invert__()
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(name='x', shape=[2, 3, 5], dtype='int32')
                b = ~x
                c = x.bitwise_not()
                d = x.__invert__()
                (b_np, c_np, d_np) = exe.run(
                    main_program,
                    feed={"x": x_np},
                    fetch_list=[b, c, d],
                )
                np.testing.assert_array_equal(res_np_b, b_np)
                np.testing.assert_array_equal(res_np_c, c_np)
                np.testing.assert_array_equal(res_np_d, d_np)

    def test_bitwise_xor(self):
        paddle.disable_static()
        x_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        y_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        res_np_b = x_np ^ y_np
        res_np_c = paddle.bitwise_xor(
            paddle.to_tensor(x_np), paddle.to_tensor(y_np)
        )
        res_np_d = x_np.__xor__(y_np)
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(name="x", shape=[2, 3, 5], dtype="int32")
                y = paddle.static.data(name="y", shape=[2, 3, 5], dtype="int32")
                b = x ^ y
                c = x.bitwise_xor(y)
                d = x.__xor__(y)
                (b_np, c_np, d_np) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[b, c, d],
                )
                np.testing.assert_array_equal(res_np_b, b_np)
                np.testing.assert_array_equal(res_np_c, c_np)
                np.testing.assert_array_equal(res_np_d, d_np)

    def test_bitwise_or(self):
        paddle.disable_static()
        x_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        y_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        res_np_b = x_np | y_np
        res_np_c = paddle.bitwise_or(
            paddle.to_tensor(x_np), paddle.to_tensor(y_np)
        )
        res_np_d = x_np.__or__(y_np)
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(name="x", shape=[2, 3, 5], dtype="int32")
                y = paddle.static.data(name="y", shape=[2, 3, 5], dtype="int32")
                b = x | y
                c = x.bitwise_or(y)
                d = x.__or__(y)
                (b_np, c_np, d_np) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[b, c, d],
                )
                np.testing.assert_array_equal(res_np_b, b_np)
                np.testing.assert_array_equal(res_np_c, c_np)
                np.testing.assert_array_equal(res_np_d, d_np)

    def test_bitwise_and(self):
        paddle.disable_static()
        x_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        y_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        res_np_b = x_np & y_np
        res_np_c = paddle.bitwise_and(
            paddle.to_tensor(x_np), paddle.to_tensor(y_np)
        )
        res_np_d = x_np.__and__(y_np)
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(name="x", shape=[2, 3, 5], dtype="int32")
                y = paddle.static.data(name="y", shape=[2, 3, 5], dtype="int32")
                b = x & y
                c = x.bitwise_and(y)
                d = x.__and__(y)
                (b_np, c_np, d_np) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[b, c, d],
                )
                np.testing.assert_array_equal(res_np_b, b_np)
                np.testing.assert_array_equal(res_np_c, c_np)
                np.testing.assert_array_equal(res_np_d, d_np)

    # for logical compare
    def test_equal_and_nequal(self):
        paddle.disable_static()
        x_np = np.array([3, 4, 10, 14, 9, 18]).astype('float32')
        y_np = np.array([3, 4, 11, 15, 8, 18]).astype('float32')
        # TODO(gouzil): Open after deleting c++ logic
        # res_np_b = x_np == y_np
        # res_np_c = paddle.equal(paddle.to_tensor(x_np), paddle.to_tensor(y_np))
        # res_np_d = x_np.__eq__(y_np)
        res_np_e = x_np != y_np
        res_np_f = paddle.not_equal(
            paddle.to_tensor(x_np), paddle.to_tensor(y_np)
        )
        res_np_g = x_np.__ne__(y_np)
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(name="x", shape=[-1, 1], dtype='float32')
                y = paddle.static.data(name="y", shape=[-1, 1], dtype='float32')
                # b = x == y
                # c = x.equal(y)
                # d = x.__eq__(y)
                e = x != y
                f = x.not_equal(y)
                g = x.__ne__(y)
                (e_np, f_np, g_np) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[e, f, g],
                )
                # np.testing.assert_array_equal(res_np_b, b_np)
                # np.testing.assert_array_equal(res_np_c, c_np)
                # np.testing.assert_array_equal(res_np_d, d_np)
                np.testing.assert_array_equal(res_np_e, e_np)
                np.testing.assert_array_equal(res_np_f, f_np)
                np.testing.assert_array_equal(res_np_g, g_np)

    def test_less(self):
        paddle.disable_static()
        x_np = np.array([3, 4, 10, 14, 9, 18]).astype('float32')
        y_np = np.array([3, 4, 11, 15, 8, 18]).astype('float32')
        z_np = np.array([3, 4, 10, 14, 9, 18]).astype('float32')
        res_np_b = x_np < y_np
        res_np_c = paddle.less_than(
            paddle.to_tensor(x_np), paddle.to_tensor(y_np)
        )
        res_np_d = x_np.__lt__(y_np)
        res_np_e = x_np <= y_np
        res_np_f = paddle.less_equal(
            paddle.to_tensor(x_np), paddle.to_tensor(y_np)
        )
        res_np_g = x_np.__le__(y_np)
        res_np_h = x_np <= z_np
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(name="x", shape=[-1, 1], dtype='float32')
                y = paddle.static.data(name="y", shape=[-1, 1], dtype='float32')
                z = paddle.static.data(name="z", shape=[-1, 1], dtype='float32')
                b = x < y
                c = x.less_than(y)
                d = x.__lt__(y)
                e = x <= y
                f = x.less_equal(y)
                g = x.__le__(y)
                h = x <= z
                (b_np, c_np, d_np, e_np, f_np, g_np, h_np) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np, "z": z_np},
                    fetch_list=[b, c, d, e, f, g, h],
                )
                np.testing.assert_array_equal(res_np_b, b_np)
                np.testing.assert_array_equal(res_np_c, c_np)
                np.testing.assert_array_equal(res_np_d, d_np)
                np.testing.assert_array_equal(res_np_e, e_np)
                np.testing.assert_array_equal(res_np_f, f_np)
                np.testing.assert_array_equal(res_np_g, g_np)
                np.testing.assert_array_equal(res_np_h, h_np)

    def test_greater(self):
        paddle.disable_static()
        x_np = np.array([3, 4, 10, 14, 9, 18]).astype('float32')
        y_np = np.array([3, 4, 11, 15, 8, 18]).astype('float32')
        z_np = np.array([3, 4, 10, 14, 9, 18]).astype('float32')
        res_np_b = x_np > y_np
        res_np_c = paddle.greater_than(
            paddle.to_tensor(x_np), paddle.to_tensor(y_np)
        )
        res_np_d = x_np.__gt__(y_np)
        res_np_e = x_np >= y_np
        res_np_f = paddle.greater_equal(
            paddle.to_tensor(x_np), paddle.to_tensor(y_np)
        )
        res_np_g = x_np.__ge__(y_np)
        res_np_h = x_np >= z_np
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(name="x", shape=[-1, 1], dtype='float32')
                y = paddle.static.data(name="y", shape=[-1, 1], dtype='float32')
                z = paddle.static.data(name="z", shape=[-1, 1], dtype='float32')
                b = x > y
                c = x.greater_than(y)
                d = x.__gt__(y)
                e = x >= y
                f = x.greater_equal(y)
                g = x.__ge__(y)
                h = x >= z
                (b_np, c_np, d_np, e_np, f_np, g_np, h_np) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np, "z": z_np},
                    fetch_list=[b, c, d, e, f, g, h],
                )
                np.testing.assert_array_equal(res_np_b, b_np)
                np.testing.assert_array_equal(res_np_c, c_np)
                np.testing.assert_array_equal(res_np_d, d_np)
                np.testing.assert_array_equal(res_np_e, e_np)
                np.testing.assert_array_equal(res_np_f, f_np)
                np.testing.assert_array_equal(res_np_g, g_np)
                np.testing.assert_array_equal(res_np_h, h_np)

    def test_item(self):
        with paddle.pir_utils.IrGuard():
            x = paddle.static.data(name='x', shape=[3, 2, 1])
            y = paddle.static.data(
                name='y',
                shape=[
                    3,
                ],
            )
            self.assertTrue(y.item().is_same(y))
            with self.assertRaises(TypeError):
                x.item()

    def test_place(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with paddle.pir_utils.IrGuard():
                x = paddle.static.data(name='x', shape=[3, 2, 1])
                _ = x.place
                self.assertTrue(len(w) == 1)
                self.assertTrue("place" in str(w[-1].message))

    def test_cpu(self):
        with paddle.pir_utils.IrGuard():
            x = paddle.static.data(name='x', shape=[3, 2, 1])
            x.cpu()

    def test_cuda(self):
        if base.is_compiled_with_cuda():
            paddle.device.set_device("gpu")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with paddle.pir_utils.IrGuard():
                    x = paddle.static.data(name='x', shape=[3, 2, 1])
                    x.cpu()
                    x.cuda(1, False)
                    self.assertTrue(len(w) == 2)
                    self.assertTrue(
                        "device_id is not supported" in str(w[-2].message)
                    )
                    self.assertTrue(
                        "blocking is not supported" in str(w[-1].message)
                    )
            paddle.device.set_device("cpu")

    def test_some_dim(self):
        with paddle.pir_utils.IrGuard():
            x = paddle.static.data(name='x', shape=[3, 2, 1])
            self.assertEqual(x.dim(), 3)
            self.assertEqual(x.ndimension(), 3)
            self.assertEqual(x.ndim, 3)

    def test_size(self):
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.assign(np.random.rand(2, 3, 4).astype("float32"))
                (output_x,) = exe.run(main_program, fetch_list=[x.size])
                self.assertEqual(output_x, 24)

    def test_T(self):
        with paddle.pir_utils.IrGuard():
            for ndim in range(5):
                # shape is [], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]
                shape = list(range(1, ndim + 1))
                out_shape = list(reversed(shape))
                main_program, exe, program_guard = new_program()
                with program_guard:
                    x = paddle.rand(shape, dtype="float32")
                    x_T = x.T
                    self.assertEqual(x_T.shape, out_shape)
                    (output_x,) = exe.run(main_program, fetch_list=[x_T])
                    self.assertEqual(output_x.shape, tuple(out_shape))

    def test_mT(self):
        with paddle.pir_utils.IrGuard():
            shape = [1]
            x = paddle.rand(shape, dtype="float32")
            self.assertRaises(ValueError, getattr, x, 'mT')

            for ndim in range(2, 5):
                # shape is [1, 2], [1, 2, 3], [1, 2, 3, 4]
                shape = list(range(1, ndim + 1))
                out_shape = list(shape)
                out_shape[-2], out_shape[-1] = out_shape[-1], out_shape[-2]
                main_program, exe, program_guard = new_program()
                with program_guard:
                    x = paddle.rand(shape, dtype="float32")
                    x_mT = x.mT
                    self.assertEqual(x_mT.shape, out_shape)
                    (output_x,) = exe.run(main_program, fetch_list=[x_mT])
                    self.assertEqual(output_x.shape, tuple(out_shape))

            shape = [1, 2, 3, 0, 1]
            out_shape = list(shape)
            out_shape[-2], out_shape[-1] = out_shape[-1], out_shape[-2]
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.rand(shape, dtype="float32")
                x_mT = x.mT
                self.assertEqual(x_mT.shape, out_shape)
                (output_x,) = exe.run(main_program, fetch_list=[x_mT])
                self.assertEqual(output_x.shape, tuple(out_shape))

            shape = [1, 2, 3, 1, 0]
            out_shape = list(shape)
            out_shape[-2], out_shape[-1] = out_shape[-1], out_shape[-2]
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.rand(shape, dtype="float32")
                x_mT = x.mT
                self.assertEqual(x_mT.shape, out_shape)
                (output_x,) = exe.run(main_program, fetch_list=[x_mT])
                self.assertEqual(output_x.shape, tuple(out_shape))

            shape = [1, 2, 3, 0, 0]
            out_shape = list(shape)
            out_shape[-2], out_shape[-1] = out_shape[-1], out_shape[-2]
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.rand(shape, dtype="float32")
                x_mT = x.mT
                self.assertEqual(x_mT.shape, out_shape)
                (output_x,) = exe.run(main_program, fetch_list=[x_mT])
                self.assertEqual(output_x.shape, tuple(out_shape))

            shape = [0, 2, 3, 0, 0]
            out_shape = list(shape)
            out_shape[-2], out_shape[-1] = out_shape[-1], out_shape[-2]
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.rand(shape, dtype="float32")
                x_mT = x.mT
                self.assertEqual(x_mT.shape, out_shape)
                (output_x,) = exe.run(main_program, fetch_list=[x_mT])
                self.assertEqual(output_x.shape, tuple(out_shape))

    def test_hash(self):
        with paddle.pir_utils.IrGuard():
            _, _, program_guard = new_program()
            with program_guard:
                x = paddle.static.data('x', [2, 3])
                self.assertEqual(hash(x), hash(id(x)))

    def test_clone(self):
        x_np = np.random.random(size=[100, 10]).astype('float64')
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(
                    name='x', shape=[100, 10], dtype="float64"
                )
                a = x.clone()
                (a_np,) = exe.run(
                    main_program,
                    feed={"x": x_np},
                    fetch_list=[a],
                )
                np.testing.assert_array_equal(x_np, a_np)
                self.assertNotEqual(id(x), id(a))

    def test_append_pop(self):
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                item_1 = paddle.static.data(
                    name='item_1', shape=[2, 3], dtype="float32"
                )
                item_2 = paddle.static.data(
                    name='item_2', shape=[3, 3], dtype="float32"
                )
                item_3 = paddle.static.data(
                    name='item_3', shape=[4, 3], dtype="float32"
                )

                item_1_np = np.random.random(size=[2, 3]).astype('float32')
                item_2_np = np.random.random(size=[3, 3]).astype('float32')
                item_3_np = np.random.random(size=[4, 3]).astype('float32')

                array = paddle.tensor.create_array('float32')
                array.append(item_1)
                array.append(item_2)
                array.append(item_3)

                sliced_item_1 = array[0]
                poped_item_3 = array.pop()
                final_length = paddle.tensor.array_length(array)
                (
                    sliced_item_1_out,
                    poped_item_3_out,
                    final_length_out,
                ) = exe.run(
                    main_program,
                    feed={
                        "item_1": item_1_np,
                        "item_2": item_2_np,
                        "item_3": item_3_np,
                    },
                    fetch_list=[sliced_item_1, poped_item_3, final_length],
                )

                np.testing.assert_array_equal(sliced_item_1_out, item_1_np)
                np.testing.assert_array_equal(poped_item_3_out, item_3_np)
                np.testing.assert_array_equal(final_length_out.item(), 2)

                with self.assertRaises(TypeError):
                    item_1.append(array)

    def test_neg(self):
        x_np = np.random.uniform(-1, 1, [10, 1024]).astype(np.float32)
        res = -x_np
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(
                    name='x', shape=[10, 1024], dtype="float32"
                )
                a = -x
                b = x.__neg__()
                (a_np, b_np) = exe.run(
                    main_program,
                    feed={"x": x_np},
                    fetch_list=[a, b],
                )
                np.testing.assert_array_equal(res, a_np)
                np.testing.assert_array_equal(res, b_np)

    def test_abs(self):
        # test for real number
        x_np = np.random.uniform(-1, 1, [10, 1024]).astype(np.float32)
        res = abs(x_np)
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(
                    name='x', shape=[10, 1024], dtype="float32"
                )
                a = abs(x)
                b = x.__abs__()
                (a_np, b_np) = exe.run(
                    main_program,
                    feed={"x": x_np},
                    fetch_list=[a, b],
                )
                np.testing.assert_array_equal(res, a_np)
                np.testing.assert_array_equal(res, b_np)

    def test_abs_complex(self):
        # test for complex number
        x_np = np.random.uniform(-1, 1, [10, 1024]).astype(
            np.float32
        ) + 1j * np.random.uniform(-1, 1, [10, 1024]).astype(np.float32)
        res = abs(x_np)
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(
                    name='x', shape=[10, 1024], dtype="complex64"
                )
                a = abs(x)
                b = x.__abs__()
                (a_np, b_np) = exe.run(
                    main_program,
                    feed={"x": x_np},
                    fetch_list=[a, b],
                )
                np.testing.assert_allclose(res, a_np, rtol=2e-7, atol=0.0)
                np.testing.assert_allclose(res, b_np, rtol=2e-7, atol=0.0)

    def test_builtin_type_conversion(self):
        with paddle.pir_utils.IrGuard():
            _, _, program_guard = new_program()
            with program_guard:
                x = paddle.static.data(name='x', shape=[], dtype="float32")
                with self.assertRaises(TypeError):
                    int(x)
                with self.assertRaises(TypeError):
                    float(x)
                with self.assertRaises(TypeError):
                    bool(x)

    def test_math_exists(self):
        with paddle.pir_utils.IrGuard():
            a = paddle.static.data(name='a', shape=[1], dtype='float32')
            self.assertTrue(isinstance(a, paddle.pir.Value))
            self.assertTrue(inspect.ismethod(a.dot))
            self.assertTrue(inspect.ismethod(a.logsumexp))
            self.assertTrue(inspect.ismethod(a.multiplex))
            self.assertTrue(inspect.ismethod(a.prod))
            self.assertTrue(inspect.ismethod(a.scale))
            self.assertTrue(inspect.ismethod(a.stanh))
            self.assertTrue(inspect.ismethod(a.add_n))
            self.assertTrue(inspect.ismethod(a.max))
            self.assertTrue(inspect.ismethod(a.maximum))
            self.assertTrue(inspect.ismethod(a.min))
            self.assertTrue(inspect.ismethod(a.minimum))
            self.assertTrue(inspect.ismethod(a.floor_divide))
            self.assertTrue(inspect.ismethod(a.remainder))
            self.assertTrue(inspect.ismethod(a.floor_mod))
            self.assertTrue(inspect.ismethod(a.multiply))
            self.assertTrue(inspect.ismethod(a.inverse))
            self.assertTrue(inspect.ismethod(a.log1p))
            self.assertTrue(inspect.ismethod(a.erf))
            self.assertTrue(inspect.ismethod(a.addmm))
            self.assertTrue(inspect.ismethod(a.clip))
            self.assertTrue(inspect.ismethod(a.trace))
            self.assertTrue(inspect.ismethod(a.kron))
            self.assertTrue(inspect.ismethod(a.isinf))
            self.assertTrue(inspect.ismethod(a.isnan))
            self.assertTrue(inspect.ismethod(a.concat))
            self.assertTrue(inspect.ismethod(a.broadcast_to))
            self.assertTrue(inspect.ismethod(a.scatter_nd_add))
            self.assertTrue(inspect.ismethod(a.scatter_nd))
            self.assertTrue(inspect.ismethod(a.shard_index))
            self.assertTrue(inspect.ismethod(a.chunk))
            self.assertTrue(inspect.ismethod(a.stack))
            self.assertTrue(inspect.ismethod(a.strided_slice))
            self.assertTrue(inspect.ismethod(a.unsqueeze))
            self.assertTrue(inspect.ismethod(a.unstack))
            self.assertTrue(inspect.ismethod(a.argmax))
            self.assertTrue(inspect.ismethod(a.argmin))
            self.assertTrue(inspect.ismethod(a.argsort))
            self.assertTrue(inspect.ismethod(a.masked_select))
            self.assertTrue(inspect.ismethod(a.topk))
            self.assertTrue(inspect.ismethod(a.index_select))
            self.assertTrue(inspect.ismethod(a.nonzero))
            self.assertTrue(inspect.ismethod(a.sort))
            self.assertTrue(inspect.ismethod(a.index_sample))
            self.assertTrue(inspect.ismethod(a.mean))
            self.assertTrue(inspect.ismethod(a.std))
            self.assertTrue(inspect.ismethod(a.numel))
            self.assertTrue(inspect.ismethod(a.asin_))
            self.assertTrue(inspect.ismethod(a.atan2))
            self.assertTrue(inspect.ismethod(a.atanh_))
            self.assertTrue(inspect.ismethod(a.diagflat))
            self.assertTrue(inspect.ismethod(a.multinomial))
            self.assertTrue(inspect.ismethod(a.pinv))
            self.assertTrue(inspect.ismethod(a.renorm))
            self.assertTrue(inspect.ismethod(a.renorm_))
            self.assertTrue(inspect.ismethod(a.tan))
            self.assertTrue(inspect.ismethod(a.tan_))
            self.assertTrue(inspect.ismethod(a.tril))
            self.assertTrue(inspect.ismethod(a.tril_))
            self.assertTrue(inspect.ismethod(a.triu))
            self.assertTrue(inspect.ismethod(a.triu_))
            self.assertTrue(inspect.ismethod(a.stft))
            self.assertTrue(inspect.ismethod(a.istft))
            self.assertTrue(inspect.ismethod(a.abs_))
            self.assertTrue(inspect.ismethod(a.acos_))
            self.assertTrue(inspect.ismethod(a.atan_))
            self.assertTrue(inspect.ismethod(a.cos_))
            self.assertTrue(inspect.ismethod(a.cosh_))
            self.assertTrue(inspect.ismethod(a.sin_))
            self.assertTrue(inspect.ismethod(a.sinh_))
            self.assertTrue(inspect.ismethod(a.acosh_))
            self.assertTrue(inspect.ismethod(a.asinh_))
            self.assertTrue(inspect.ismethod(a.diag))

    def test_binary_op_with_scalar(self):
        with paddle.pir_utils.IrGuard():
            main_program, exe, program_guard = new_program()
            with program_guard:
                x_np = np.array(10, dtype=np.int32)
                x = paddle.static.data(name='x', shape=[], dtype="int32")
                y1 = x / 2
                y2 = x / 5.0
                y3 = x // 2
                y4 = x * 8.0
                self.assertEqual(y1.dtype, paddle.pir.core.DataType.FLOAT32)
                self.assertEqual(y2.dtype, paddle.pir.core.DataType.FLOAT32)
                self.assertEqual(y3.dtype, paddle.pir.core.DataType.INT32)
                self.assertEqual(y4.dtype, paddle.pir.core.DataType.FLOAT32)
                (y1_out, y2_out, y3_out, y4_out) = exe.run(
                    main_program,
                    feed={
                        "x": x_np,
                    },
                    fetch_list=[y1, y2, y3, y4],
                )
                np.testing.assert_allclose(x_np / 2, y1_out, rtol=1e-05)
                np.testing.assert_allclose(x_np / 5.0, y2_out, rtol=1e-05)
                np.testing.assert_allclose(x_np // 2, y3_out, atol=1e-05)
                np.testing.assert_allclose(x_np * 8.0, y4_out, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()

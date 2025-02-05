# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
from paddle import base


class TestRepeatInterleaveOp(OpTest):
    def setUp(self):
        self.op_type = "repeat_interleave"
        self.python_api = paddle.repeat_interleave
        self.init_dtype_type()
        index_np = np.random.randint(
            low=0, high=3, size=self.index_size
        ).astype(self.index_type)
        x_np = np.random.random(self.x_shape).astype(self.x_type)

        self.inputs = {'X': x_np, 'RepeatsTensor': index_np}
        self.attrs = {'dim': self.dim}

        outer_loop = np.prod(self.x_shape[: self.dim])
        x_reshape = [outer_loop, *self.x_shape[self.dim :]]
        x_np_reshape = np.reshape(x_np, tuple(x_reshape))
        out_list = []
        for i in range(outer_loop):
            for j in range(self.index_size):
                for k in range(index_np[j]):
                    out_list.append(x_np_reshape[i, j])
        self.out_shape = list(self.x_shape)
        self.out_shape[self.dim] = np.sum(index_np)
        self.out_shape = tuple(self.out_shape)

        out = np.reshape(out_list, self.out_shape)
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        self.dim = 1
        self.x_type = np.float64
        self.index_type = np.int64
        self.x_shape = (8, 4, 5)
        self.index_size = self.x_shape[self.dim]

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', check_pir=True)


class TestRepeatInterleaveOp2(OpTest):
    def setUp(self):
        self.op_type = "repeat_interleave"
        self.python_api = paddle.repeat_interleave
        self.init_dtype_type()
        index_np = 2
        x_np = np.random.random(self.x_shape).astype(self.x_type)
        self.inputs = {'X': x_np}  # , 'RepeatsTensor': None}
        self.attrs = {'dim': self.dim, 'Repeats': index_np}

        outer_loop = np.prod(self.x_shape[: self.dim])
        x_reshape = [outer_loop, *self.x_shape[self.dim :]]
        x_np_reshape = np.reshape(x_np, tuple(x_reshape))
        out_list = []
        for i in range(outer_loop):
            for j in range(self.index_size):
                for k in range(index_np):
                    out_list.append(x_np_reshape[i, j])
        self.out_shape = list(self.x_shape)
        self.out_shape[self.dim] = index_np * self.index_size
        self.out_shape = tuple(self.out_shape)

        out = np.reshape(out_list, self.out_shape)
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        self.dim = 1
        self.x_type = np.float64
        self.x_shape = (8, 4, 5)
        self.index_size = self.x_shape[self.dim]

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', check_pir=True)


class TestIndexSelectAPI(unittest.TestCase):
    def input_data(self):
        self.data_zero_dim_x = np.array(0.5).astype('float32')
        self.data_x = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        ).astype('float32')
        self.data_zero_dim_index = np.array(2)
        self.data_index = np.array([0, 1, 2, 1]).astype('int32')

    def test_repeat_interleave_api(self):
        paddle.enable_static()
        self.input_data()

        # case 1:
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
            index = paddle.static.data(
                name='repeats_',
                shape=[4],
                dtype='int32',
            )
            if not paddle.framework.in_pir_mode():
                x.desc.set_need_check_feed(False)
                index.desc.set_need_check_feed(False)
            x.stop_gradient = False
            index.stop_gradient = False
            z = paddle.repeat_interleave(x, index, axis=1)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': self.data_x, 'repeats_': self.data_index},
                fetch_list=[z],
                return_numpy=False,
            )
        expect_out = np.repeat(self.data_x, self.data_index, axis=1)
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        # case 2:
        repeats = np.array([1, 2, 1]).astype('int32')
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='x', shape=[-1, 4], dtype="float32")
            index = paddle.static.data(
                name='repeats_',
                shape=[3],
                dtype='int32',
            )
            if not paddle.framework.in_pir_mode():
                x.desc.set_need_check_feed(False)
                index.desc.set_need_check_feed(False)
            z = paddle.repeat_interleave(x, index, axis=0)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={
                    'x': self.data_x,
                    'repeats_': repeats,
                },
                fetch_list=[z],
                return_numpy=False,
            )
        expect_out = np.repeat(self.data_x, repeats, axis=0)
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        repeats = 2
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
            z = paddle.repeat_interleave(x, repeats, axis=0)
            if not paddle.framework.in_pir_mode():
                x.desc.set_need_check_feed(False)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': self.data_x}, fetch_list=[z], return_numpy=False
            )
        expect_out = np.repeat(self.data_x, repeats, axis=0)
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        # case 3 zero_dim:
        if not paddle.framework.in_pir_mode():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[-1], dtype="float32")
                if not paddle.framework.in_pir_mode():
                    x.desc.set_need_check_feed(False)
                z = paddle.repeat_interleave(x, repeats)
                exe = base.Executor(base.CPUPlace())
                (res,) = exe.run(
                    feed={'x': self.data_zero_dim_x},
                    fetch_list=[z],
                    return_numpy=False,
                )
            expect_out = np.repeat(self.data_zero_dim_x, repeats)
            np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        # case 4 negative axis:
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
            index = paddle.static.data(
                name='repeats_',
                shape=[4],
                dtype='int32',
            )

            if not paddle.framework.in_pir_mode():
                x.desc.set_need_check_feed(False)
                index.desc.set_need_check_feed(False)
            z = paddle.repeat_interleave(x, index, axis=-1)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': self.data_x, 'repeats_': self.data_index},
                fetch_list=[z],
                return_numpy=False,
            )
        expect_out = np.repeat(self.data_x, self.data_index, axis=-1)
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_dygraph_api(self):
        self.input_data()
        # case axis none
        input_x = np.array([[1, 2, 1], [1, 2, 3]]).astype('int32')
        index_x = np.array([1, 1, 2, 1, 2, 2]).astype('int32')

        with base.dygraph.guard():
            x = paddle.to_tensor(input_x)
            index = paddle.to_tensor(index_x)
            z = paddle.repeat_interleave(x, index, None)
            np_z = z.numpy()
        expect_out = np.repeat(input_x, index_x, axis=None)
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case repeats int
        with base.dygraph.guard():
            x = paddle.to_tensor(input_x)
            index = 2
            z = paddle.repeat_interleave(x, index, None)
            np_z = z.numpy()
        expect_out = np.repeat(input_x, index, axis=None)
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case input dtype is bfloat16
        input_x = np.array([[1, 2, 1], [1, 2, 3]]).astype('uint16')

        with base.dygraph.guard():
            x = paddle.to_tensor(input_x)
            index = paddle.to_tensor(index_x)
            z = paddle.repeat_interleave(x, index, None)
            np_z = z.numpy()
        expect_out = np.repeat(input_x, index_x, axis=None)
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        with base.dygraph.guard():
            x = paddle.to_tensor(input_x)
            index = 2
            z = paddle.repeat_interleave(x, index, None)
            np_z = z.numpy()
        expect_out = np.repeat(input_x, index, axis=None)
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case 1:
        with base.dygraph.guard():
            x = paddle.to_tensor(self.data_x)
            index = paddle.to_tensor(self.data_index)
            z = paddle.repeat_interleave(x, index, -1)
            np_z = z.numpy()
        expect_out = np.repeat(self.data_x, self.data_index, axis=-1)
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        with base.dygraph.guard():
            x = paddle.to_tensor(self.data_x)
            index = paddle.to_tensor(self.data_index)
            z = paddle.repeat_interleave(x, index, 1)
            np_z = z.numpy()
        expect_out = np.repeat(self.data_x, self.data_index, axis=1)
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case 2:
        index_x = np.array([1, 2, 1]).astype('int32')
        with base.dygraph.guard():
            x = paddle.to_tensor(self.data_x)
            index = paddle.to_tensor(index_x)
            z = paddle.repeat_interleave(x, index, axis=0)
            np_z = z.numpy()
        expect_out = np.repeat(self.data_x, index, axis=0)
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case 3 zero_dim:
        with base.dygraph.guard():
            x = paddle.to_tensor(self.data_zero_dim_x)
            index = 2
            z = paddle.repeat_interleave(x, index, None)
            np_z = z.numpy()
        expect_out = np.repeat(self.data_zero_dim_x, index, axis=None)
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case 4 zero_dim_index
        with base.dygraph.guard():
            x = paddle.to_tensor(self.data_zero_dim_x)
            index = paddle.to_tensor(self.data_zero_dim_index)
            z = paddle.repeat_interleave(x, index, None)
            np_z = z.numpy()
        expect_out = np.repeat(
            self.data_zero_dim_x, self.data_zero_dim_index, axis=None
        )
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()

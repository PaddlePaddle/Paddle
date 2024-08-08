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

import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle import base


def ref_repeat_interleave(x_np, index_np, axis):
    x_shape = x_np.shape
    if axis < 0:
        axis += len(x_shape)
    index_size = x_shape[axis]
    if not isinstance(index_np, np.ndarray):
        index_np = np.full([index_size], index_np, dtype=np.int32)

    outer_loop = np.prod(x_shape[:axis])
    x_reshape = [outer_loop, *x_shape[axis:]]
    x_np_reshape = np.reshape(x_np, tuple(x_reshape))
    out_list = []
    for i in range(outer_loop):
        for j in range(index_size):
            for k in range(index_np[j]):
                out_list.append(x_np_reshape[i, j])
    out_shape = list(x_shape)
    out_shape[axis] = np.sum(index_np)
    out_shape = tuple(out_shape)

    out = np.reshape(out_list, out_shape)
    return out


class XPUTestRepeatInterleaveOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "repeat_interleave"

    class TestRepeatInterleaveOp(XPUOpTest):
        def setUp(self):
            self.op_type = "repeat_interleave"
            self.python_api = paddle.repeat_interleave

            self.init_case()
            x_np = np.random.random(self.x_shape).astype(self.x_type)
            self.inputs = {'X': x_np}
            self.attrs = {'dim': self.dim}
            if hasattr(self, "index") and self.index is not None:
                index_np = self.index
                self.attrs['Repeats'] = index_np
            else:
                index_np = np.random.randint(
                    low=0, high=5, size=self.x_shape[self.dim]
                ).astype(self.index_type)
                self.inputs['RepeatsTensor'] = index_np

            out = ref_repeat_interleave(x_np, index_np, self.dim)
            self.outputs = {'Out': out}

        def init_case(self):
            self.dim = 1
            self.x_type = self.in_type
            self.index_type = np.int64
            self.x_shape = (8, 4, 5)

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

        def test_check_grad(self):
            place = paddle.XPUPlace(0)
            self.check_grad(place, ['X'], 'Out')

    class TestRepeatInterleaveOp2(TestRepeatInterleaveOp):
        def init_case(self):
            self.dim = 1
            self.x_type = self.in_type
            self.x_shape = (8, 4, 5)
            self.index = 2


support_types = get_xpu_op_support_types('repeat_interleave')
for stype in support_types:
    create_test_class(globals(), XPUTestRepeatInterleaveOp, stype)


class TestRepeatInterleaveAPI(unittest.TestCase):
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
            exe = base.Executor(base.XPUPlace(0))
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
            exe = base.Executor(base.XPUPlace(0))
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
            exe = base.Executor(base.XPUPlace(0))
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
                exe = base.Executor(base.XPUPlace(0))
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
            exe = base.Executor(base.XPUPlace(0))
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

#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, get_places

import paddle
from paddle import base
from paddle.base import core


def one_hot_wrapper(x, depth_tensor, **keargs):
    return paddle.nn.functional.one_hot(x, depth_tensor)


class TestOneHotOp(OpTest):
    def setUp(self):
        self.op_type = 'one_hot_v2'
        self.prim_op_type = "comp"
        self.python_api = one_hot_wrapper
        self.public_python_api = one_hot_wrapper
        self.python_out_sig = ['Out']
        depth = 10
        depth_np = np.array(10).astype('int32')
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype('int32').reshape([sum(x_lod[0])])

        out = np.zeros(shape=(np.prod(x.shape), depth)).astype('float32')

        for i in range(np.prod(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {'X': x, 'depth_tensor': depth_np}
        self.attrs = {'dtype': int(core.VarDesc.VarType.FP32)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_cinn=True, check_prim_pir=True)


class TestOneHotOp_dims(OpTest):
    def setUp(self):
        self.op_type = 'one_hot_v2'
        self.prim_op_type = "comp"
        self.python_api = one_hot_wrapper
        self.public_python_api = one_hot_wrapper
        self.python_out_sig = ['Out']
        depth = 10
        depth_np = np.array(10).astype('int32')
        x_shape = [5, 10, 7, 3]
        x = [np.random.randint(0, depth - 1) for i in range(np.prod(x_shape))]
        x = np.array(x).astype('int32').reshape(x_shape)

        out = np.zeros(shape=(np.prod(x.shape), depth)).astype('float32')

        r_x = np.reshape(x, np.prod(x.shape))
        for i in range(np.prod(x.shape)):
            out[i, r_x[i]] = 1.0

        shape_np = list(x.shape)
        shape_np.append(depth)
        out = np.reshape(out, shape_np)

        self.inputs = {'X': x, 'depth_tensor': depth_np}
        self.attrs = {'dtype': int(core.VarDesc.VarType.FP32)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_cinn=True, check_prim_pir=True)


class TestOneHotOp_attr(OpTest):
    def setUp(self):
        self.op_type = 'one_hot_v2'
        self.prim_op_type = "comp"
        self.python_api = one_hot_wrapper
        self.public_python_api = one_hot_wrapper
        depth = 10
        depth_np = np.array(10).astype('int32')
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype('int32').reshape([sum(x_lod[0]), 1])

        out = np.zeros(shape=(np.prod(x.shape[:-1]), 1, depth)).astype(
            'float32'
        )

        for i in range(np.prod(x.shape)):
            out[i, 0, x[i]] = 1.0

        self.inputs = {'X': x, 'depth_tensor': depth_np}
        self.attrs = {'dtype': int(core.VarDesc.VarType.FP32), 'depth': depth}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_cinn=True, check_prim_pir=True)


class TestOneHotOp_default_dtype(OpTest):
    def setUp(self):
        self.op_type = 'one_hot_v2'
        self.prim_op_type = "comp"
        self.python_api = one_hot_wrapper
        self.public_python_api = one_hot_wrapper
        depth = 10
        depth_np = np.array(10).astype('int32')
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype('int32').reshape([sum(x_lod[0])])

        out = np.zeros(shape=(np.prod(x.shape), depth)).astype('float32')

        for i in range(np.prod(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {'X': x, 'depth_tensor': depth_np}
        self.attrs = {}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_cinn=True, check_prim_pir=True)


class TestOneHotOp_default_dtype_attr(OpTest):
    def setUp(self):
        self.op_type = 'one_hot_v2'
        self.prim_op_type = "comp"
        self.python_api = one_hot_wrapper
        self.public_python_api = one_hot_wrapper
        depth = 10
        depth_np = np.array(depth).astype('int32')
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype('int32').reshape([sum(x_lod[0]), 1])

        out = np.zeros(shape=(np.prod(x.shape[:-1]), 1, depth)).astype(
            'float32'
        )

        for i in range(np.prod(x.shape)):
            out[i, 0, x[i]] = 1.0

        self.inputs = {'X': x}
        self.attrs = {'depth': depth}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_cinn=True, check_prim_pir=True)


class TestOneHotOpApi(unittest.TestCase):
    def test_api(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            depth = 10
            label = paddle.static.data(
                name="label", shape=[-1, 1], dtype="int64"
            )
            one_hot_label = paddle.nn.functional.one_hot(
                x=label, num_classes=depth
            )

            place = paddle.CPUPlace()
            label_data = np.array(
                [np.random.randint(0, 10 - 1) for i in range(6)]
            ).reshape([6, 1])
            label_data = label_data.astype('int64')

            exe = base.Executor(place)
            exe.run(startup)
            ret = exe.run(
                feed={
                    'label': label_data,
                },
                fetch_list=[one_hot_label],
                return_numpy=False,
            )

    def test_api_with_depthTensor(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            depth = paddle.assign(np.array([10], dtype=np.int32))
            label = paddle.static.data(
                name="label", shape=[-1, 1], dtype="int64"
            )
            one_hot_label = paddle.nn.functional.one_hot(
                x=label, num_classes=depth
            )

            place = paddle.CPUPlace()
            label_data = np.array(
                [np.random.randint(0, 10 - 1) for i in range(6)]
            ).reshape([6, 1])
            label_data = label_data.astype('int64')

            exe = base.Executor(place)
            exe.run(startup)
            ret = exe.run(
                feed={
                    'label': label_data,
                },
                fetch_list=[one_hot_label],
                return_numpy=False,
            )

    def test_api_with_dygraph(self):
        depth = 10
        label = np.array(
            [np.random.randint(0, depth - 1) for i in range(6)]
        ).reshape([6, 1])
        with base.dygraph.guard():
            one_hot_label = paddle.nn.functional.one_hot(
                paddle.to_tensor(label), depth
            )


class BadInputTestOnehotV2(unittest.TestCase):
    def test_error(self):
        with paddle.static.program_guard(paddle.static.Program()):

            def test_bad_x():
                label = paddle.static.data(
                    name="label",
                    shape=[-1, 4],
                    dtype="float32",
                )
                if not paddle.framework.use_pir_api():
                    label.desc.need_check_feed(False)
                one_hot_label = paddle.nn.functional.one_hot(
                    x=label, num_classes=4
                )

            self.assertRaises(TypeError, test_bad_x)


class TestOneHotOp_ZeroSize(OpTest):
    def setUp(self):
        self.op_type = 'one_hot_v2'
        self.python_api = one_hot_wrapper
        self.public_python_api = one_hot_wrapper
        self.python_out_sig = ['Out']
        depth = 10
        depth_np = np.array(10).astype('int32')
        x_shape = [0, 10, 7, 3]
        x = [np.random.randint(0, depth - 1) for i in range(np.prod(x_shape))]
        x = np.array(x).astype('int32').reshape(x_shape)

        out = np.zeros(shape=(np.prod(x.shape), depth)).astype('float32')

        r_x = np.reshape(x, np.prod(x.shape))
        for i in range(np.prod(x.shape)):
            out[i, r_x[i]] = 1.0

        shape_np = list(x.shape)
        shape_np.append(depth)
        out = np.reshape(out, shape_np)

        self.inputs = {'X': x, 'depth_tensor': depth_np}
        self.attrs = {'dtype': int(core.VarDesc.VarType.FP32)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()


class TestOneHotAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.places = get_places()
        self.shape = [5]
        self.dtype = 'int32'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.randint(0, 8, self.shape).astype(self.dtype)
        self.num_classes = self.np_input.max() + 1
        self.np_out = np.eye(self.num_classes)[self.np_input]

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        paddle_dygraph_out = []
        # Position args (args)
        out1 = paddle.nn.functional.one_hot(x, self.num_classes)
        paddle_dygraph_out.append(out1)
        # Keywords args (kwargs) for paddle
        out2 = paddle.nn.functional.one_hot(x=x, num_classes=self.num_classes)
        paddle_dygraph_out.append(out2)
        # Keywords args for torch
        out3 = paddle.nn.functional.one_hot(
            input=x, num_classes=self.num_classes
        )
        paddle_dygraph_out.append(out3)
        # default args
        out4 = paddle.nn.functional.one_hot(x, -1)
        paddle_dygraph_out.append(out4)
        # Check
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(self.np_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            # Position args (args)
            out1 = paddle.nn.functional.one_hot(x, self.num_classes)
            # Keywords args (kwargs) for paddle
            out2 = paddle.nn.functional.one_hot(
                x=x, num_classes=self.num_classes
            )
            # Keywords args for torch
            out3 = paddle.nn.functional.one_hot(
                input=x, num_classes=self.num_classes
            )
            # default args
            out4 = paddle.nn.functional.one_hot(x, -1)
            exe = base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_input},
                fetch_list=[out1, out2, out3],
            )
            for out in fetches:
                np.testing.assert_allclose(out, self.np_out)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

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
from op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
from utils import static_guard

import paddle
from paddle import base
from paddle.base import core


class TestGatherNdOpWithEmptyIndex(OpTest):
    # Index has empty element, which means copy entire tensor

    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        self.if_enable_cinn()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.random((5, 20)).astype(target_dtype)
        output = np.vstack((xnp[np.newaxis, :], xnp[np.newaxis, :]))
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': np.array([[], []]).astype("int32")}
        self.outputs = {'Out': output}

    def if_enable_cinn(self):
        pass

    def config_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestGatherNdOpWithEmptyIndexFP16(TestGatherNdOpWithEmptyIndex):
    def config_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestGatherNdOpWithEmptyIndexBF16(TestGatherNdOpWithEmptyIndex):
    def config_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestGatherNdOpWithIndex1(OpTest):
    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        self.if_enable_cinn()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.random((5, 20)).astype(target_dtype)
        index = np.array([1]).astype("int32")
        output = xnp[index[-1]]
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': index}
        self.outputs = {'Out': output}

    def if_enable_cinn(self):
        pass

    def config_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestGatherNdOpWithIndex1_ZeroDim(TestGatherNdOpWithIndex1):
    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        self.if_enable_cinn()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.random((100,)).astype(target_dtype)
        index = np.array([1]).astype("int32")
        output = xnp[index[-1]]
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': index}
        self.outputs = {'Out': output}

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestGatherNdOpWithIndex1FP16(TestGatherNdOpWithIndex1):
    def config_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestGatherNdOpWithIndex1BF16(TestGatherNdOpWithIndex1):
    def config_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestGatherNdOpWithLowIndex(OpTest):
    # Index has low rank, X has high rank

    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.uniform(0, 100, (10, 10)).astype(target_dtype)
        index = np.array([[1], [2]]).astype("int64")
        output = xnp[tuple(index.T)]  # shape is [2, 10]

        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)

        self.inputs = {'X': xnp, 'Index': index}
        self.outputs = {'Out': output}

    def config_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestGatherNdOpWithLowIndexFP16(TestGatherNdOpWithLowIndex):
    def config_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestGatherNdOpWithLowIndexBF16(TestGatherNdOpWithLowIndex):
    def config_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            numeric_grad_delta=0.5,
            check_prim_pir=True,
        )


class TestGatherNdOpIndex1(OpTest):
    # Index has low rank, X has high rank

    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.uniform(0, 100, (10, 10)).astype(target_dtype)
        if self.dtype == np.uint16:
            xnp = convert_uint16_to_float(convert_float_to_uint16(xnp))
        index = np.array([1, 2]).astype("int32")
        output = xnp[tuple(index.T)]
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': index}
        self.outputs = {'Out': output}
        self.if_enable_cinn()

    def if_enable_cinn(self):
        # the outputs are 0D-tensor, CINN not support
        self.enable_cinn = False

    def config_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            numeric_grad_delta=0.05,
            check_prim_pir=True,
        )


class TestGatherNdOpIndex1FP16(TestGatherNdOpIndex1):
    def config_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestGatherNdOpIndex1BF16(TestGatherNdOpIndex1):
    def config_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            numeric_grad_delta=0.5,
            check_prim_pir=True,
        )


class TestGatherNdOpWithSameIndexAsX(OpTest):
    # Index has same rank as X's rank
    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.uniform(0, 100, (10, 10)).astype(target_dtype)
        index = np.array([[1, 1], [2, 1]]).astype("int64")
        output = xnp[tuple(index.T)]  # [25, 22]
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': index}
        self.outputs = {'Out': output}

    def config_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestGatherNdOpWithSameIndexAsXFP16(TestGatherNdOpWithSameIndexAsX):
    def config_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestGatherNdOpWithSameIndexAsXBF16(TestGatherNdOpWithSameIndexAsX):
    def config_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            numeric_grad_delta=0.5,
            check_prim_pir=True,
        )


class TestGatherNdOpWithHighRankSame(OpTest):
    # Both Index and X have high rank, and Rank(Index) = Rank(X)

    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        shape = (5, 2, 3, 1, 10)
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.rand(*shape).astype(target_dtype)
        index = np.vstack([np.random.randint(-s, s, size=2) for s in shape]).T
        output = xnp[tuple(index.T)]
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': index.astype("int32")}
        self.outputs = {'Out': output}

    def config_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestGatherNdOpWithHighRankSameFP16(TestGatherNdOpWithHighRankSame):
    def config_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestGatherNdOpWithHighRankSameBF16(TestGatherNdOpWithHighRankSame):
    def config_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestGatherNdOpWithHighRankDiff(OpTest):
    # Both Index and X have high rank, and Rank(Index) < Rank(X)

    def setUp(self):
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        shape = (2, 3, 4, 1, 10)
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.rand(*shape).astype(target_dtype)
        index = np.vstack([np.random.randint(-s, s, size=200) for s in shape]).T
        index_re = index.reshape([20, 5, 2, 5])
        output = xnp[tuple(index.T)].reshape([20, 5, 2])
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': index_re.astype("int32")}
        self.outputs = {'Out': output}

    def config_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestGatherNdOpWithHighRankDiffFP16(TestGatherNdOpWithHighRankDiff):
    def config_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestGatherNdOpWithHighRankDiffBF16(TestGatherNdOpWithHighRankDiff):
    def config_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


# Test Python API
class TestGatherNdOpAPI(unittest.TestCase):

    def test_case1(self):
        with static_guard():
            x1 = paddle.static.data(
                name='x1', shape=[-1, 30, 40, 50, 60], dtype='float32'
            )
            index1 = paddle.static.data(
                name='index1', shape=[-1, 2, 4], dtype='int32'
            )
            output1 = paddle.gather_nd(x1, index1)

    def test_case2(self):
        with static_guard():
            x2 = paddle.static.data(
                name='x2', shape=[-1, 30, 40, 50], dtype='float32'
            )
            index2 = paddle.static.data(
                name='index2', shape=[-1, 2, 2], dtype='int64'
            )
            output2 = paddle.gather_nd(x2, index2)

    def test_case3(self):
        with static_guard():
            x3 = paddle.static.data(
                name='x3', shape=[-1, 3, 4, 5], dtype='float32'
            )
            index3 = paddle.static.data(
                name='index3', shape=[-1, 2, 1], dtype='int32'
            )
            output3 = paddle.gather_nd(x3, index3, name="gather_nd_layer")


# Test Raise Index Error
class TestGatherNdOpRaise(unittest.TestCase):

    def test_check_raise(self):
        def check_raise_is_test():
            with static_guard():
                try:
                    x = paddle.static.data(
                        name='x', shape=[-1, 3, 4, 5], dtype='float32'
                    )
                    index = paddle.static.data(
                        name='index', shape=[-1, 2, 10], dtype='int32'
                    )
                    output = paddle.gather_nd(x, index)
                except Exception as e:
                    t = "Input(Index).shape[-1] should be no greater than Input(X).rank"
                    if t in str(e):
                        raise IndexError

        self.assertRaises(IndexError, check_raise_is_test)


class TestGatherNdError(unittest.TestCase):

    def test_error1(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                shape = [8, 9, 6]
                x = paddle.static.data(shape=shape, dtype='float32', name='x')
                index = paddle.static.data(
                    shape=shape, dtype='bool', name='index'
                )
                np_x = np.random.random(shape).astype('float32')
                np_index = np.array(
                    np.random.randint(2, size=shape, dtype=bool)
                )

                def test_x_type():
                    paddle.gather_nd(np_x, index)

                self.assertRaises(TypeError, test_x_type)

                def test_index_type():
                    paddle.gather_nd(x, np_index)

                self.assertRaises(TypeError, test_index_type)

    def test_error2(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                shape = [8, 9, 6]
                x = paddle.static.data(shape=shape, dtype='float32', name='x')
                index_float = paddle.static.data(
                    shape=shape, dtype='float32', name='index_float'
                )

                def test_index_dtype():
                    paddle.gather_nd(x, index_float)

                self.assertRaises(TypeError, test_index_dtype)


class TestGatherNdAPI2(unittest.TestCase):

    def test_static(self):
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data('data1', shape=[-1, 2], dtype='float64')
            index = paddle.static.data('index', shape=[-1, 1], dtype='int32')
            out = paddle.gather_nd(data1, index)
            place = base.CPUPlace()
            exe = base.Executor(place)
            input = np.array([[1, 2], [3, 4], [5, 6]]).astype('float64')
            index_1 = np.array([[1]]).astype('int32')
            (result,) = exe.run(
                feed={"data1": input, "index": index_1}, fetch_list=[out]
            )
            expected_output = np.array([[3, 4]])
        np.testing.assert_allclose(result, expected_output, rtol=1e-05)

    def test_static_fp16_with_gpu(self):
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = np.array(
                    [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]],
                    dtype='float16',
                )
                index = np.array([[0, 1]], dtype='int32')
                res_np = np.array([[3, 4]], dtype='float16')

                x = paddle.static.data(
                    name="x", shape=[2, 3, 2], dtype="float16"
                )
                idx = paddle.static.data(
                    name="index", shape=[1, 2], dtype="int32"
                )

                y = paddle.gather_nd(x, idx)

                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": input, "index": index},
                    fetch_list=[y],
                )

                np.testing.assert_allclose(res[0], res_np, rtol=1e-05)

    def test_imperative(self):
        paddle.disable_static()
        input_1 = np.array([[1, 2], [3, 4], [5, 6]])
        index_1 = np.array([[1]])
        input = paddle.to_tensor(input_1)
        index = paddle.to_tensor(index_1)
        output = paddle.gather(input, index)
        output_np = output.numpy()
        expected_output = np.array([[3, 4]])
        np.testing.assert_allclose(output_np, expected_output, rtol=1e-05)
        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()

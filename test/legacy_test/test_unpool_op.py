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

import os
import unittest

import numpy as np
from op_test import OpTest
from test_attribute_var import UnittestBase

import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.framework import in_pir_mode


def _unpool_output_size(x, kernel_size, stride, padding, output_size):
    input_size = x.shape
    default_size = []
    for d in range(len(kernel_size)):
        default_size.append(
            (input_size[-len(kernel_size) + d] - 1) * stride[d]
            + kernel_size[d]
            - 2 * padding[d]
        )
    if output_size is None:
        ret = default_size
    else:
        ret = output_size
    return ret


def unpool2dmax_forward_naive(
    input, indices, ksize, strides, paddings, output_size
):
    s0, s1, s2, s3 = input.shape
    output_size = _unpool_output_size(
        input, ksize, strides, paddings, output_size
    )
    out_hsize = output_size[0]
    out_wsize = output_size[1]
    out = np.zeros((s0, s1, out_hsize, out_wsize))
    for nidx in range(s0):
        for cidx in range(s1):
            for h in range(s2):
                for w in range(s3):
                    index = indices[nidx, cidx, h, w]
                    hidx = (index - index % out_wsize) // out_wsize
                    widx = index % out_wsize
                    out[nidx, cidx, hidx, widx] = input[nidx, cidx, h, w]

    return out


def max_unpool2d_wrapper(
    x,
    indices,
    kernel_size,
    stride=None,
    padding=0,
    output_size=None,
    data_format="NCHW",
    name=None,
):
    out = paddle.nn.functional.max_unpool2d(
        x,
        indices,
        kernel_size,
        stride=stride,
        padding=padding,
        data_format=data_format,
        output_size=output_size,
        name=name,
    )
    return out


class TestUnpoolOp(OpTest):
    def setUp(self):
        self.op_type = "unpool"
        self.python_api = max_unpool2d_wrapper
        self.init_test_case()
        input = np.random.randint(0, 100, self.shape)
        nsize, csize, hsize, wsize = input.shape
        self.output_size = _unpool_output_size(
            input, self.ksize, self.strides, self.paddings, self.output_size
        )
        indices = np.random.permutation(
            np.arange(0, self.output_size[0] * self.output_size[1])
        )[: hsize * wsize]
        indices = np.reshape(indices, [hsize, wsize])
        idx_list = []
        for n in range(nsize):
            c_list = []
            for c in range(csize):
                c_list.append(indices.tolist())
            idx_list.append(c_list)
        indices = np.array(idx_list)

        output = self.unpool2d_forward_naive(
            input,
            indices,
            self.ksize,
            self.strides,
            self.paddings,
            self.output_size,
        ).astype("float64")

        self.inputs = {
            'X': input.astype('float64'),
            'Indices': indices.astype('int32'),
        }
        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'unpooling_type': self.unpooling_type,
            'output_size': self.output_size,
        }
        self.outputs = {'Out': output.astype('float64')}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)

    def init_test_case(self):
        self.unpool2d_forward_naive = unpool2dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [2, 4, 7, 8]
        self.ksize = [2, 2]
        self.strides = [2, 2]
        self.paddings = [0, 0]
        self.output_size = None


class TestUnpoolOpcase1(TestUnpoolOp):
    def init_test_case(self):
        self.unpool2d_forward_naive = unpool2dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [3, 2, 5, 5]
        self.ksize = [4, 4]
        self.strides = [2, 2]
        self.paddings = [0, 0]
        self.output_size = None


class TestUnpoolOpOutputsize(TestUnpoolOp):
    def init_test_case(self):
        self.unpool2d_forward_naive = unpool2dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [3, 2, 5, 5]
        self.ksize = [4, 4]
        self.strides = [2, 2]
        self.paddings = [0, 0]
        self.output_size = [12, 12]


class TestUnpoolOpOutput(TestUnpoolOp):
    def init_test_case(self):
        self.unpool2d_forward_naive = unpool2dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [3, 2, 5, 5]
        self.ksize = [4, 4]
        self.strides = [2, 2]
        self.paddings = [0, 0]
        self.output_size = [12, 12]


class TestUnpoolOpException(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_exception(self):
        def indices_size_error():
            data = paddle.rand(shape=[1, 1, 3, 3])
            indices = paddle.reshape(
                paddle.arange(0, 12), shape=[1, 1, 3, 4]
            ).astype("int32")
            F.max_unpool2d(data, indices, kernel_size=2, stride=2)

        def x_rank_error():
            data = paddle.rand(shape=[1, 1, 3])
            indices = paddle.reshape(
                paddle.arange(0, 9), shape=[1, 1, 3, 3]
            ).astype("int32")
            F.max_unpool2d(data, indices, kernel_size=2, stride=2)

        def indices_rank_error():
            data = paddle.rand(shape=[1, 1, 3, 3])
            indices = paddle.reshape(
                paddle.arange(0, 9), shape=[1, 3, 3]
            ).astype("int32")
            F.max_unpool2d(data, indices, kernel_size=2, stride=2)

        def indices_value_error():
            data = paddle.rand(shape=[1, 1, 3, 3])
            indices = paddle.reshape(
                paddle.arange(31, 40), shape=[1, 1, 3, 3]
            ).astype("int32")
            F.max_unpool2d(data, indices, kernel_size=2, stride=2)

        def data_format_error():
            data = paddle.rand(shape=[1, 1, 3, 3])
            indices = paddle.reshape(
                paddle.arange(0, 9), shape=[1, 1, 3, 3]
            ).astype("int32")
            F.max_unpool2d(
                data, indices, kernel_size=2, stride=2, data_format="NHWC"
            )

        def data_outputsize_error():
            data = paddle.rand(shape=[1, 1, 3, 3])
            indices = paddle.reshape(
                paddle.arange(0, 9), shape=[1, 1, 3, 3]
            ).astype("int32")
            F.max_unpool2d(
                data, indices, kernel_size=2, stride=2, output_size=[5, 6, 7, 8]
            )

        def data_outputsize_error2():
            data = paddle.rand(shape=[1, 1, 3, 3])
            indices = paddle.reshape(
                paddle.arange(0, 9), shape=[1, 1, 3, 3]
            ).astype("int32")
            F.max_unpool2d(
                data, indices, kernel_size=2, stride=2, output_size=[100, 100]
            )

        self.assertRaisesRegex(
            ValueError,
            r"The dimensions of Input\(X\) must equal to",
            indices_size_error,
        )
        self.assertRaisesRegex(
            ValueError,
            r"The x should have \[N, C, H, W\] format",
            x_rank_error,
        )
        self.assertRaisesRegex(
            ValueError,
            r"The indices should have \[N, C, H, W\] format",
            indices_rank_error,
        )
        if not core.is_compiled_with_cuda():
            self.assertRaisesRegex(
                ValueError,
                r"index should less than output",
                indices_value_error,
            )
        self.assertRaisesRegex(
            ValueError,
            r"Attr\(data_format\) should be 'NCHW'",
            data_format_error,
        )
        self.assertRaisesRegex(
            ValueError, r"invalid output_size", data_outputsize_error
        )
        self.assertRaisesRegex(
            ValueError, r"invalid output_size", data_outputsize_error2
        )


class TestUnpoolOpAPI_dy(unittest.TestCase):
    def test_case(self):
        import numpy as np

        import paddle
        import paddle.nn.functional as F
        from paddle import base
        from paddle.base import core

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with base.dygraph.guard(place):
            input_data = np.array(
                [
                    [
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ]
                    ]
                ]
            ).astype("float32")
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool2d(
                input_x, kernel_size=2, stride=2, return_mask=True
            )
            out_pp = F.max_unpool2d(
                output, indices, kernel_size=2, stride=2, output_size=(5, 5)
            )
            output_np = output.numpy()
            indices_np = indices.numpy()
            expect_res = unpool2dmax_forward_naive(
                output_np, indices_np, [2, 2], [2, 2], [0, 0], [5, 5]
            ).astype("float64")
            np.testing.assert_allclose(out_pp.numpy(), expect_res, rtol=1e-05)


class TestUnpoolOpAPI_dy2(unittest.TestCase):
    def test_case(self):
        import numpy as np

        import paddle
        import paddle.nn.functional as F
        from paddle import base
        from paddle.base import core

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with base.dygraph.guard(place):
            input_data = np.array(
                [
                    [
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ]
                    ]
                ]
            ).astype("float32")
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool2d(
                input_x, kernel_size=2, stride=2, return_mask=True
            )
            out_pp = F.max_unpool2d(
                output, indices, kernel_size=2, stride=None, output_size=(5, 5)
            )
            output_np = output.numpy()
            indices_np = indices.numpy()
            expect_res = unpool2dmax_forward_naive(
                output_np, indices_np, [2, 2], [2, 2], [0, 0], [5, 5]
            ).astype("float64")
            np.testing.assert_allclose(out_pp.numpy(), expect_res, rtol=1e-05)


class TestUnpoolOpAPI_dy3(unittest.TestCase):
    def test_case(self):
        import numpy as np

        import paddle
        from paddle import base
        from paddle.base import core

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with base.dygraph.guard(place):
            input_data = np.array(
                [
                    [
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ]
                    ]
                ]
            ).astype("float32")
            input_x = paddle.to_tensor(input_data)
            Pool2d = paddle.nn.MaxPool2D(
                kernel_size=2, stride=2, return_mask=True
            )
            UnPool = paddle.nn.MaxUnPool2D(kernel_size=2, stride=2)

            output, indices = Pool2d(input_x)
            out_pp = UnPool(output, indices)
            output_np = output.numpy()
            indices_np = indices.numpy()
            expect_res = unpool2dmax_forward_naive(
                output_np, indices_np, [2, 2], [2, 2], [0, 0], [4, 4]
            ).astype("float64")
            np.testing.assert_allclose(out_pp.numpy(), expect_res, rtol=1e-05)


class TestUnpoolOpAPI_dy4(unittest.TestCase):
    def test_case(self):
        import numpy as np

        import paddle
        import paddle.nn.functional as F
        from paddle import base
        from paddle.base import core

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with base.dygraph.guard(place):
            input_data = np.array(
                [
                    [
                        [
                            [1, 2, 3, 4, 5],
                            [6, 7, 8, 9, 10],
                            [11, 12, 13, 14, 15],
                            [16, 17, 18, 19, 20],
                        ]
                    ]
                ]
            ).astype("float32")
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool2d(
                input_x, kernel_size=2, stride=2, return_mask=True
            )
            out_pp = F.max_unpool2d(
                output.astype("int64"),
                indices,
                kernel_size=2,
                stride=None,
                output_size=input_x.shape,
            )
            output_np = output.numpy()
            indices_np = indices.numpy()
            expect_res = unpool2dmax_forward_naive(
                output_np, indices_np, [2, 2], [2, 2], [0, 0], [4, 5]
            ).astype("float64")
            np.testing.assert_allclose(out_pp.numpy(), expect_res, rtol=1e-05)


class TestUnpoolOpAPI_st(unittest.TestCase):

    def test_case(self):
        import paddle
        import paddle.nn.functional as F
        from paddle.base import core

        paddle.enable_static()

        input_data = np.array(
            [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]
        ).astype("float32")

        x = paddle.static.data(name="x", shape=[1, 1, 4, 4], dtype="float32")
        output, indices = F.max_pool2d(
            x, kernel_size=2, stride=2, return_mask=True
        )
        unpool_out = F.max_unpool2d(
            output, indices, kernel_size=2, stride=None, output_size=(5, 5)
        )
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = paddle.static.Executor(place)

        results = exe.run(
            feed={"x": input_data},
            fetch_list=[unpool_out],
            return_numpy=True,
        )

        pool_out_np = np.array([[[[6.0, 8.0], [14.0, 16.0]]]]).astype("float32")
        indices_np = np.array([[[[5, 7], [13, 15]]]]).astype("int32")
        expect_res = unpool2dmax_forward_naive(
            pool_out_np, indices_np, [2, 2], [2, 2], [0, 0], [5, 5]
        ).astype("float64")
        np.testing.assert_allclose(results[0], expect_res, rtol=1e-05)
        paddle.disable_static()


class TestOutputSizeTensor(UnittestBase):
    def init_info(self):
        self.shapes = [[1, 3, 6, 6]]
        self.save_path = os.path.join(self.temp_dir.name, self.path_prefix())

    def test_static(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            fc = paddle.nn.Linear(6, 6)
            x = paddle.randn(self.shapes[0])
            x.stop_gradient = False
            feat = fc(x)  # [1,3,6,6]

            out = self.call_func(feat)

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))

            if not in_pir_mode():
                self.assertTrue(self.var_prefix() in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(startup_prog)
            res = exe.run(fetch_list=[out])
            np.testing.assert_array_equal(res[0].shape, [1, 3, 7, 7])

            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            # Test for Inference Predictor
            infer_outs = self.infer_prog()
            np.testing.assert_array_equal(res[0].shape, [1, 3, 7, 7])

    def path_prefix(self):
        return 'unpool_var'

    def var_prefix(self):
        return "Vars["

    def call_func(self, x):
        output_size = [paddle.assign([7]), paddle.assign([7])]
        pool_out, indices = F.max_pool2d(
            x, kernel_size=2, stride=2, padding=0, return_mask=True
        )
        # pool_out shape: [1, 1, 6, 6],  indices shape: [1, 1, 6, 6]
        unpool_out = F.max_unpool2d(
            pool_out, indices, kernel_size=2, padding=0, output_size=output_size
        )
        # unpool_out shape: [1, 1, 7, 7]
        return unpool_out


class TestZOutputSizeTensor2(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_dygraph(self):
        x = paddle.randn([1, 3, 6, 6])
        pool_out, indices = F.max_pool2d(
            x, kernel_size=2, stride=2, padding=0, return_mask=True
        )
        output_size = [paddle.assign([7]), paddle.assign([7])]
        unpool_out = F.max_unpool2d(
            pool_out, indices, kernel_size=2, padding=0, output_size=output_size
        )
        np.testing.assert_array_equal(unpool_out.shape, [1, 3, 7, 7])


class TestZOutputSizeTensor3(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_dygraph(self):
        x = paddle.randn([1, 3, 6, 6])
        pool_out, indices = F.max_pool2d(
            x, kernel_size=2, stride=2, padding=0, return_mask=True
        )
        output_size = [
            paddle.assign([1]),
            paddle.assign([1]),
            paddle.assign([7]),
            paddle.assign([7]),
        ]
        unpool_out = F.max_unpool2d(
            pool_out, indices, kernel_size=2, padding=0, output_size=output_size
        )
        np.testing.assert_array_equal(unpool_out.shape, [1, 3, 7, 7])


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

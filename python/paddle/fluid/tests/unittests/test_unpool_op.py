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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest


def _unpool_output_size(x, kernel_size, stride, padding, output_size):
    input_size = x.shape
    default_size = []
    for d in range(len(kernel_size)):
        default_size.append((input_size[-len(kernel_size) + d] - 1) * stride[d]
                            + kernel_size[d] - 2 * padding[d])
    if output_size is None:
        ret = default_size
    else:
        ret = output_size
    return ret


def unpool2dmax_forward_naive(input, indices, ksize, strides, paddings,
                              output_size):
    s0, s1, s2, s3 = input.shape
    output_size = _unpool_output_size(input, ksize, strides, paddings,
                                      output_size)
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
                    out[nidx, cidx, hidx, widx] = \
                            input[nidx, cidx, h, w]

    return out


class TestUnpoolOp(OpTest):
    def setUp(self):
        self.op_type = "unpool"
        self.init_test_case()
        input = np.random.randint(0, 100, self.shape)
        nsize, csize, hsize, wsize = input.shape
        self.output_size = _unpool_output_size(input, self.ksize, self.strides,
                                               self.paddings, self.output_size)
        indices = np.random.permutation(
            np.arange(0, self.output_size[0] * self.output_size[1]))[:hsize *
                                                                     wsize]
        indices = np.reshape(indices, [hsize, wsize])
        idx_list = []
        for n in range(nsize):
            c_list = []
            for c in range(csize):
                c_list.append(indices.tolist())
            idx_list.append(c_list)
        indices = np.array(idx_list)

        output = self.unpool2d_forward_naive(input, indices, self.ksize, \
                self.strides, self.paddings, self.output_size).astype("float64")

        self.inputs = {
            'X': input.astype('float64'),
            'Indices': indices.astype('int32')
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
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

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


class TestUnpoolOpOuputsize(TestUnpoolOp):
    def init_test_case(self):
        self.unpool2d_forward_naive = unpool2dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [3, 2, 5, 5]
        self.ksize = [4, 4]
        self.strides = [2, 2]
        self.paddings = [0, 0]
        self.output_size = [9, 9]


class TestUnpoolOpOuput(TestUnpoolOp):
    def init_test_case(self):
        self.unpool2d_forward_naive = unpool2dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [3, 2, 5, 5]
        self.ksize = [4, 4]
        self.strides = [2, 2]
        self.paddings = [0, 0]
        self.output_size = [9, 9]


class TestUnpoolOpException(unittest.TestCase):
    def test_exception(self):
        import paddle.nn.functional as F
        import paddle

        def indices_size_error():
            data = paddle.randint(shape=[1, 1, 3, 3])
            indices = paddle.reshape(paddle.arange(0, 12), shape[1, 1, 3, 4])
            MaxPool2D = F.maxunpool2d(data, indices, kernel_size=2, stride=2)

        def indices_value_error():
            data = paddle.randint(shape=[1, 1, 3, 3])
            indices = paddle.reshape(paddle.arange(4, 40), shape[1, 1, 3, 4])
            MaxPool2D = F.maxunpool2d(data, indices, kernel_size=2, stride=2)

        def data_format_error():
            data = paddle.randint(shape=[1, 1, 3, 3])
            indices = paddle.reshape(paddle.arange(4, 40), shape[1, 1, 3, 4])
            MaxPool2D = F.maxunpool2d(
                data, indices, kernel_size=2, stride=2, data_format="NHWC")

        def data_outputsize_error():
            data = paddle.randint(shape=[1, 1, 3, 3])
            indices = paddle.reshape(paddle.arange(4, 40), shape[1, 1, 3, 4])
            MaxPool2D = F.maxunpool2d(
                data,
                indices,
                kernel_size=2,
                stride=2,
                output_size=[5, 6, 7, 8])

        def data_outputsize_error2():
            data = paddle.randint(shape=[1, 1, 3, 3])
            indices = paddle.reshape(paddle.arange(4, 40), shape[1, 1, 3, 4])
            MaxPool2D = F.maxunpool2d(
                data, indices, kernel_size=2, stride=2, output_size=[100, 100])

        self.assertRaises(ValueError, indices_size_error)
        self.assertRaises(ValueError, indices_value_error)
        self.assertRaises(ValueError, data_format_error)
        self.assertRaises(ValueError, data_outputsize_error)
        self.assertRaises(ValueError, data_outputsize_error2)


class TestUnpoolOpAPI_dy(unittest.TestCase):
    def test_case(self):
        import paddle
        import paddle.nn.functional as F
        import paddle.fluid.core as core
        import paddle.fluid as fluid
        import numpy as np

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with fluid.dygraph.guard(place):
            input_data = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8],
                                     [9, 10, 11, 12],
                                     [13, 14, 15, 16]]]]).astype("float32")
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool2d(
                input_x, kernel_size=2, stride=2, return_mask=True)
            out_pp = F.max_unpool2d(
                output, indices, kernel_size=2, stride=2, output_size=(5, 5))
            output_np = output.numpy()
            indices_np = indices.numpy()
            expect_res =unpool2dmax_forward_naive(output_np, indices_np, [2,2], \
                [2,2], [0,0], [5,5]).astype("float64")
            self.assertTrue(np.allclose(out_pp.numpy(), expect_res))


class TestUnpoolOpAPI_dy2(unittest.TestCase):
    def test_case(self):
        import paddle
        import paddle.nn.functional as F
        import paddle.fluid.core as core
        import paddle.fluid as fluid
        import numpy as np

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with fluid.dygraph.guard(place):
            input_data = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8],
                                     [9, 10, 11, 12],
                                     [13, 14, 15, 16]]]]).astype("float32")
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool2d(
                input_x, kernel_size=2, stride=2, return_mask=True)
            out_pp = F.max_unpool2d(
                output, indices, kernel_size=2, stride=None, output_size=(5, 5))
            output_np = output.numpy()
            indices_np = indices.numpy()
            expect_res =unpool2dmax_forward_naive(output_np, indices_np, [2,2], \
                [2,2], [0,0], [5,5]).astype("float64")
            self.assertTrue(np.allclose(out_pp.numpy(), expect_res))


class TestUnpoolOpAPI_dy3(unittest.TestCase):
    def test_case(self):
        import paddle
        import paddle.nn.functional as F
        import paddle.fluid.core as core
        import paddle.fluid as fluid
        import numpy as np

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with fluid.dygraph.guard(place):
            input_data = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8],
                                     [9, 10, 11, 12],
                                     [13, 14, 15, 16]]]]).astype("float32")
            input_x = paddle.to_tensor(input_data)
            Pool2d = paddle.nn.MaxPool2D(
                kernel_size=2, stride=2, return_mask=True)
            UnPool = paddle.nn.MaxUnPool2D(kernel_size=2, stride=2)

            output, indices = Pool2d(input_x)
            out_pp = UnPool(output, indices)
            output_np = output.numpy()
            indices_np = indices.numpy()
            expect_res =unpool2dmax_forward_naive(output_np, indices_np, [2,2], \
                [2,2], [0,0], [4,4]).astype("float64")
            self.assertTrue(np.allclose(out_pp.numpy(), expect_res))


class TestUnpoolOpAPI_st(unittest.TestCase):
    def test_case(self):
        import paddle
        import paddle.nn.functional as F
        import paddle.fluid.core as core
        import paddle.fluid as fluid
        paddle.enable_static()

        input_data = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                                 [13, 14, 15, 16]]]]).astype("float32")

        x = fluid.data(name="x", shape=[1, 1, 4, 4], dtype="float32")
        output, indices = F.max_pool2d(
            x, kernel_size=2, stride=2, return_mask=True)
        unpool_out = F.max_unpool2d(
            output, indices, kernel_size=2, stride=None, output_size=(5, 5))
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        results = exe.run(paddle.fluid.default_main_program(),\
                          feed={"x":input_data},
                          fetch_list=[unpool_out],
                          return_numpy=True)

        pool_out_np = np.array([[[[6., 8.], [14., 16.]]]]).astype("float32")
        indices_np = np.array([[[[5, 7], [13, 15]]]]).astype("int32")
        expect_res =unpool2dmax_forward_naive(pool_out_np, indices_np, [2,2], \
            [2,2], [0,0], [5,5]).astype("float64")
        self.assertTrue(np.allclose(results[0], expect_res))


if __name__ == '__main__':
    unittest.main()

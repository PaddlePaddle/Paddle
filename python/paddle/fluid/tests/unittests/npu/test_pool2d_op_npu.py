#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest
import numpy as np
sys.path.append("..")

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest
from test_pool2d_op import pool2D_forward_naive, avg_pool2D_forward_naive, max_pool2D_forward_naive
from paddle.nn.functional import avg_pool2d, max_pool2d

paddle.enable_static()


def create_test_padding_SAME_class(parent):
    @unittest.skipIf(not paddle.is_compiled_with_npu(),
                     "core is not compiled with NPU")
    class TestPaddingSMAECase(parent):
        def init_paddings(self):
            self.paddings = [0, 0]
            self.padding_algorithm = "SAME"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSMAECase


def create_test_use_ceil_class(parent):
    @unittest.skipIf(not paddle.is_compiled_with_npu(),
                     "core is not compiled with NPU")
    class TestPool2DUseCeilCase(parent):
        def init_ceil_mode(self):
            self.ceil_mode = True

    cls_name = "{0}_{1}".format(parent.__name__, "CeilModeCast")
    TestPool2DUseCeilCase.__name__ = cls_name
    globals()[cls_name] = TestPool2DUseCeilCase


def create_test_padding_VALID_class(parent):
    @unittest.skipIf(not paddle.is_compiled_with_npu(),
                     "core is not compiled with NPU")
    class TestPaddingVALIDCase(parent):
        def init_paddings(self):
            self.paddings = [1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


def pool2d_backward_navie(x,
                          ksize,
                          strides,
                          padding_algorithm,
                          paddings,
                          pool_type,
                          is_global,
                          is_adaptive,
                          is_exclusive,
                          data_format="NCHW"):
    # update paddings
    def _get_padding_with_SAME(input_shape, pool_size, pool_stride):
        padding = []
        for input_size, filter_size, stride_size in zip(input_shape, pool_size,
                                                        pool_stride):
            out_size = int((input_size + stride_size - 1) / stride_size)
            pad_sum = np.max((
                (out_size - 1) * stride_size + filter_size - input_size, 0))
            pad_0 = int(pad_sum / 2)
            pad_1 = int(pad_sum - pad_0)
            padding.append(pad_0)
            padding.append(pad_1)
        return padding

    if isinstance(padding_algorithm, str):
        padding_algorithm = padding_algorithm.upper()
        if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
            raise ValueError("Unknown Attr(padding_algorithm): '%s'. "
                             "It can only be 'SAME' or 'VALID'." %
                             str(padding_algorithm))

        if padding_algorithm == "VALID":
            paddings = [0, 0, 0, 0]
        elif padding_algorithm == "SAME":
            input_data_shape = []
            if data_format == "NCHW":
                input_data_shape = x.shape[2:4]
            elif data_format == "NHWC":
                input_data_shape = x.shape[1:3]
            paddings = _get_padding_with_SAME(input_data_shape, ksize, strides)

    if data_format == "NHWC":
        x = x.transpose([0, 3, 1, 2])

    if is_global:
        ksize = x.shape[2:]
        paddings = [0, 0, 0, 0]

    if is_adaptive:
        pass

    if len(paddings) == 2:
        paddings = [paddings[0], paddings[0], paddings[1], paddings[1]]
    x_shape = np.array(x.shape) + np.array(
        [0, 0, paddings[0] + paddings[1], paddings[2] + paddings[3]])
    x_new = np.zeros(x_shape)

    x_old_shape = x.shape
    x_new_shape = x_new.shape

    N, C, H, W = x_old_shape
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    x_new[n, c, h + paddings[0], w + paddings[2]] = x[n, c, h,
                                                                      w]
    x = x_new
    N, C, H, W = x_new_shape
    x_grad = np.zeros(x.shape, dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for h in range(0, H - ksize[0] + 1, strides[0]):
                for w in range(0, W - ksize[1] + 1, strides[1]):
                    start_h = h
                    start_w = w
                    end_h = h + ksize[0]
                    end_w = w + ksize[1]
                    if is_exclusive:
                        start_h = max(start_h, paddings[0])
                        start_w = max(start_w, paddings[2])
                        end_h = min(end_h, H - paddings[1])
                        end_w = min(end_w, W - paddings[3])
                    idx = np.meshgrid(
                        range(start_h, end_h), range(start_w, end_w))
                    if pool_type == "max":
                        idx = np.argmax(x[n, c, idx[0], idx[1]].flatten())
                        idx_h = idx // (end_w - start_w)
                        idx_w = idx % (end_w - start_w)
                        x_grad[n, c, start_h + idx_h, start_w + idx_w] += 1
                    elif pool_type == "avg":
                        idx = np.meshgrid(
                            range(start_h, end_h), range(start_w, end_w))
                        x_grad[n, c, idx[0], idx[1]] += 1 / (
                            end_h - start_h) / (end_w - start_w)

    x_grad_new = np.zeros(x_old_shape)
    N, C, H, W = x_old_shape
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    x_grad_new[n, c, h, w] = x_grad[n, c, h + paddings[0], w +
                                                    paddings[2]]
    x_grad = x_grad_new
    if data_format == "NHWC":
        x_grad = x_grad.transpose([0, 2, 3, 1])
    return x_grad


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestPool2D_Op(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "pool2d"
        self.init_kernel_type()
        self.init_data_type()
        self.init_test_case()
        self.padding_algorithm = "EXPLICIT"
        self.init_paddings()
        self.init_global_pool()
        self.init_kernel_type()
        self.init_pool_type()
        self.init_ceil_mode()
        self.init_exclusive()
        self.init_adaptive()
        self.init_data_format()
        self.init_shape()

        input = np.random.random(self.shape).astype(self.dtype)
        if self.pool_type == "max":
            input = np.array([x for x in range(np.prod(self.shape))]).reshape(
                self.shape).astype(self.dtype)
        output = pool2D_forward_naive(
            input, self.ksize, self.strides, self.paddings, self.global_pool,
            self.ceil_mode, self.exclusive, self.adaptive, self.data_format,
            self.pool_type, self.padding_algorithm).astype(self.dtype)
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(input)}

        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'pooling_type': self.pool_type,
            'global_pooling': self.global_pool,
            'use_cudnn': False,
            'use_mkldnn': False,
            'ceil_mode': self.ceil_mode,
            'data_format': self.data_format,
            'exclusive': self.exclusive,
            'adaptive': self.adaptive,
            "padding_algorithm": self.padding_algorithm,
        }

        self.outputs = {'Out': output}

    def init_data_format(self):
        self.data_format = "NCHW"

    def init_shape(self):
        self.shape = [2, 3, 5, 5]

    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]

    def init_paddings(self):
        self.paddings = [0, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_kernel_type(self):
        self.use_cudnn = False

    def init_data_type(self):
        self.dtype = np.float32

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_global_pool(self):
        self.global_pool = True

    def init_ceil_mode(self):
        self.ceil_mode = False

    def init_exclusive(self):
        self.exclusive = True

    def init_adaptive(self):
        self.adaptive = False

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(fluid.NPUPlace(0), atol=1e-3)

    def test_check_grad(self):
        x_grad = pool2d_backward_navie(
            self.inputs['X'], self.ksize, self.strides, self.padding_algorithm,
            self.paddings, self.pool_type, self.global_pool, self.adaptive,
            self.exclusive, self.data_format)
        x_grad = x_grad / np.prod(self.outputs['Out'].shape)
        self.check_grad_with_place(
            fluid.NPUPlace(0),
            set(['X']),
            'Out',
            max_relative_error=0.06,
            user_defined_grads=[x_grad])


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase1(TestPool2D_Op):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]

    def init_paddings(self):
        self.paddings = [0, 0]

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_global_pool(self):
        self.global_pool = False

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase2(TestPool2D_Op):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]

    def init_paddings(self):
        self.paddings = [1, 1]

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_global_pool(self):
        self.global_pool = False

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase3(TestPool2D_Op):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase4(TestCase1):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase5(TestCase2):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAvgInclude(TestCase2):
    def init_exclusive(self):
        self.exclusive = False


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAvgPoolAdaptive(TestCase1):
    def init_adaptive(self):
        self.adaptive = True

    def init_shape(self):
        self.shape = [2, 3, 7, 7]

    def init_test_case(self):
        self.ksize = [7, 7]
        self.strides = [7, 7]
        self.paddings = [0, 0, 0, 0]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAvgPoolAdaptiveAsyOutSize(TestCase1):
    def init_adaptive(self):
        self.adaptive = True

    def init_shape(self):
        self.shape = [2, 3, 8, 8]

    def init_test_case(self):
        self.ksize = [2, 4]
        self.strides = [2, 4]
        self.paddings = [0, 0, 0, 0]


#-------test pool2d with asymmetric padding-----


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestPool2D_AsyPadding(TestPool2D_Op):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 0, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 5, 5]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase1_AsyPadding(TestCase1):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 0, 1, 0]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase2_AsyPadding(TestCase2):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 2, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase3_AsyPadding(TestCase3):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 0, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 5, 5]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase4_AsyPadding(TestCase4):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 0, 1, 0]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase5_AsyPadding((TestCase5)):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [2, 2, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAvgInclude_AsyPadding(TestCase2):
    def init_exclusive(self):
        self.exclusive = False

    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 2, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


# class TestAvgPoolAdaptive_AsyPadding(TestCase1):
#     def init_adaptive(self):
#         self.adaptive = True

#     def init_test_case(self):
#         self.ksize = [2, 2]
#         self.strides = [2, 2]
#         self.paddings = [0, 0, 0, 0]

#     def init_shape(self):
#         self.shape = [2, 3, 8, 8]


#----------- test channel_last --------------
@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestPool2D_channel_last(TestPool2D_Op):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 5, 5, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase1_channel_last(TestCase1):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase2_channel_last(TestCase2):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase3_channel_last(TestCase3):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 5, 5, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase4_channel_last(TestCase4):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase5_channel_last(TestCase5):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase5_Max(TestCase2):
    def init_pool_type(self):
        self.pool_type = "max"


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase5_channel_last_Max(TestCase5_Max):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAvgInclude_channel_last(TestCase2_channel_last):
    def init_exclusive(self):
        self.exclusive = False


# @unittest.skipIf(not paddle.is_compiled_with_npu(),
#                  "core is not compiled with NPU")
# class TestAvgPoolAdaptive_channel_last(TestCase1_channel_last):
#     def init_adaptive(self):
#         self.adaptive = True


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestPool2D_AsyPadding_channel_last(TestPool2D_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 5, 5, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase1_AsyPadding_channel_last(TestCase1_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase2_AsyPadding_channel_last(TestCase2_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase3_AsyPadding_channel_last(TestCase3_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 5, 5, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase4_AsyPadding_channel_last(TestCase4_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase5_AsyPadding_channel_last(TestCase5_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAvgInclude_AsyPadding_channel_last(TestAvgInclude_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


# @unittest.skipIf(not paddle.is_compiled_with_npu(),
#                  "core is not compiled with NPU")
# class TestAvgPoolAdaptive_AsyPadding_channel_last(
#         TestAvgPoolAdaptive_AsyPadding):
#     def init_data_format(self):
#         self.data_format = "NHWC"

#     def init_shape(self):
#         self.shape = [2, 7, 7, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase1_strides(TestCase1):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 2]

    def init_shape(self):
        self.shape = [2, 3, 4, 5]


create_test_padding_SAME_class(TestPool2D_Op)
create_test_padding_SAME_class(TestCase1)
create_test_padding_SAME_class(TestCase2)
create_test_padding_SAME_class(TestCase3)
create_test_padding_SAME_class(TestCase4)
create_test_padding_SAME_class(TestCase5)
create_test_padding_SAME_class(TestPool2D_channel_last)
create_test_padding_SAME_class(TestCase1_channel_last)
create_test_padding_SAME_class(TestCase2_channel_last)
create_test_padding_SAME_class(TestCase3_channel_last)
create_test_padding_SAME_class(TestCase4_channel_last)
create_test_padding_SAME_class(TestCase5_channel_last)
create_test_padding_SAME_class(TestCase1_strides)

create_test_padding_VALID_class(TestPool2D_Op)
create_test_padding_VALID_class(TestCase1)
create_test_padding_VALID_class(TestCase2)
create_test_padding_VALID_class(TestCase3)
create_test_padding_VALID_class(TestCase4)
create_test_padding_VALID_class(TestCase5)
create_test_padding_VALID_class(TestPool2D_channel_last)
create_test_padding_VALID_class(TestCase1_channel_last)
create_test_padding_VALID_class(TestCase2_channel_last)
create_test_padding_VALID_class(TestCase3_channel_last)
create_test_padding_VALID_class(TestCase4_channel_last)
create_test_padding_VALID_class(TestCase5_channel_last)

create_test_use_ceil_class(TestCase1)
create_test_use_ceil_class(TestCase2)
create_test_use_ceil_class(TestCase1_AsyPadding)
create_test_use_ceil_class(TestCase2_AsyPadding)
create_test_use_ceil_class(TestCase1_channel_last)
create_test_use_ceil_class(TestCase2_channel_last)
create_test_use_ceil_class(TestCase1_AsyPadding_channel_last)
create_test_use_ceil_class(TestCase2_AsyPadding_channel_last)

if __name__ == "__main__":
    unittest.main()

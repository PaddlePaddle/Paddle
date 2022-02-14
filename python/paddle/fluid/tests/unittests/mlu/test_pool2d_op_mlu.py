#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import division

import unittest
import numpy as np

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import sys
sys.path.append('..')
from op_test import OpTest
from test_pool2d_op import pool2D_forward_naive, avg_pool2D_forward_naive, max_pool2D_forward_naive


class TestPool2D_Op_Mixin(object):
    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.op_type = "pool2d"
        self.init_data_type()
        self.init_test_case()
        self.padding_algorithm = "EXPLICIT"
        self.init_paddings()
        self.init_global_pool()
        self.init_pool_type()
        self.init_ceil_mode()
        self.init_exclusive()
        self.init_adaptive()
        self.init_data_format()
        self.init_shape()

        input = np.random.random(self.shape).astype(self.dtype)
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
            'ceil_mode': self.ceil_mode,
            'data_format': self.data_format,
            'exclusive': self.exclusive,
            'adaptive': self.adaptive,
            "padding_algorithm": self.padding_algorithm,
        }

        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return

        if self.pool_type != "max":
            self.check_grad_with_place(
                self.place, set(['X']), 'Out', max_relative_error=0.07)

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

    # Not support adaptive pooling currently
    def init_adaptive(self):
        self.adaptive = False


class TestPool2D_Op(TestPool2D_Op_Mixin, OpTest):
    pass


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


class TestCase3(TestPool2D_Op):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase4(TestCase1):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase5(TestCase2):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


def create_test_fp16_class(parent, check_grad=True):
    class TestFp16Case(parent):
        def init_data_type(self):
            self.dtype = np.float16

        def test_check_output(self):
            place = core.MLUPlace(0)
            self.check_output_with_place(place, atol=1e-3)

        def test_check_grad(self):
            place = core.MLUPlace(0)
            if self.pool_type != "max" and check_grad:
                self.check_grad_with_place(
                    place, set(['X']), 'Out', max_relative_error=0.07)

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16Op")
    TestFp16Case.__name__ = cls_name
    globals()[cls_name] = TestFp16Case


create_test_fp16_class(TestPool2D_Op)
create_test_fp16_class(TestCase1, check_grad=False)
create_test_fp16_class(TestCase2)
create_test_fp16_class(TestCase3)
create_test_fp16_class(TestCase4)
create_test_fp16_class(TestCase5)

#--------------------test pool2d use ceil mode--------------------


def create_test_use_ceil_class(parent):
    class TestPool2DUseCeilCase(parent):
        def init_ceil_mode(self):
            self.ceil_mode = True

    cls_name = "{0}_{1}".format(parent.__name__, "CeilModeCast")
    TestPool2DUseCeilCase.__name__ = cls_name
    globals()[cls_name] = TestPool2DUseCeilCase


create_test_use_ceil_class(TestCase1)
create_test_use_ceil_class(TestCase2)


class TestAvgInclude(TestCase2):
    def init_exclusive(self):
        self.exclusive = False


#-------test pool2d with asymmetric padding-----


class TestPool2D_AsyPadding(TestPool2D_Op):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 0, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 5, 5]


class TestCase1_AsyPadding(TestCase1):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 0, 1, 0]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


class TestCase2_AsyPadding(TestCase2):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 2, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


class TestCase3_AsyPadding(TestCase3):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 0, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 5, 5]


class TestCase4_AsyPadding(TestCase4):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 0, 1, 0]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


class TestCase5_AsyPadding((TestCase5)):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [2, 2, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


create_test_use_ceil_class(TestCase1_AsyPadding)
create_test_use_ceil_class(TestCase2_AsyPadding)


class TestAvgInclude_AsyPadding(TestCase2):
    def init_exclusive(self):
        self.exclusive = False

    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 2, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


#----------- test channel_last --------------
class TestPool2D_channel_last(TestPool2D_Op):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 5, 5, 3]


class TestCase1_channel_last(TestCase1):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestCase2_channel_last(TestCase2):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestCase3_channel_last(TestCase3):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 5, 5, 3]


class TestCase4_channel_last(TestCase4):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestCase5_channel_last(TestCase5):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


create_test_use_ceil_class(TestCase1_channel_last)
create_test_use_ceil_class(TestCase2_channel_last)


class TestCase5_Max(TestCase2):
    def init_pool_type(self):
        self.pool_type = "max"

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        place = core.MLUPlace(0)
        if self.pool_type == "max":
            self.check_grad_with_place(
                place, set(['X']), 'Out', max_relative_error=1.00)


class TestCase5_channel_last_Max(TestCase5_Max):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestAvgInclude_channel_last(TestCase2_channel_last):
    def init_exclusive(self):
        self.exclusive = False


class TestPool2D_AsyPadding_channel_last(TestPool2D_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 5, 5, 3]


class TestCase1_AsyPadding_channel_last(TestCase1_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestCase2_AsyPadding_channel_last(TestCase2_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestCase3_AsyPadding_channel_last(TestCase3_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 5, 5, 3]


class TestCase4_AsyPadding_channel_last(TestCase4_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestCase5_AsyPadding_channel_last(TestCase5_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


create_test_use_ceil_class(TestCase1_AsyPadding_channel_last)
create_test_use_ceil_class(TestCase2_AsyPadding_channel_last)


class TestAvgInclude_AsyPadding_channel_last(TestAvgInclude_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


# test paddings: SAME VALID


def create_test_padding_SAME_class(parent):
    class TestPaddingSMAECase(parent):
        def init_paddings(self):
            self.paddings = [0, 0]
            self.padding_algorithm = "SAME"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSMAECase


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


def create_test_padding_VALID_class(parent):
    class TestPaddingVALIDCase(parent):
        def init_paddings(self):
            self.paddings = [1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


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


class TestCase1_strides(TestCase1):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 2]

    def init_shape(self):
        self.shape = [2, 3, 4, 5]


create_test_padding_SAME_class(TestCase1_strides)


# ----- test API
class TestPool2DAPI(unittest.TestCase):
    def test_api(self):
        x_NHWC = np.random.random([2, 5, 5, 3]).astype("float32")
        x_NCHW = np.random.random([2, 3, 5, 5]).astype("float32")

        input_NHWC = fluid.layers.data(
            name="input_NHWC",
            shape=[2, 5, 5, 3],
            append_batch_size=False,
            dtype="float32")

        input_NCHW = fluid.layers.data(
            name="input_NCHW",
            shape=[2, 3, 5, 5],
            append_batch_size=False,
            dtype="float32")

        input_NHWC_negetive = fluid.layers.data(
            name="input_NHWC_negetive",
            shape=[2, -1, 5, 3],
            append_batch_size=False,
            dtype="float32")

        input_NCHW_negetive = fluid.layers.data(
            name="input_NCHW_negetive",
            shape=[2, 3, -1, -1],
            append_batch_size=False,
            dtype="float32")

        ksize = [3, 3]
        out_1 = fluid.layers.pool2d(
            input=input_NHWC,
            pool_size=ksize,
            pool_type="max",
            pool_padding=[1, 1],
            data_format="NHWC")

        out_2 = fluid.layers.pool2d(
            input=input_NHWC,
            pool_size=ksize,
            pool_type="avg",
            pool_padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
            data_format="NHWC")

        out_3 = fluid.layers.pool2d(
            input=input_NCHW,
            pool_size=ksize,
            pool_type="avg",
            pool_padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
            data_format="NCHW")

        out_4 = fluid.layers.pool2d(
            input=input_NCHW,
            pool_size=ksize,
            pool_type="avg",
            pool_padding=[1, 2, 1, 0],
            data_format="NCHW")
        # test VALID
        out_5 = fluid.layers.pool2d(
            input=input_NCHW,
            pool_size=ksize,
            pool_type="avg",
            pool_padding="VALID",
            data_format="NCHW")

        out_6 = fluid.layers.pool2d(
            input=input_NHWC,
            pool_size=ksize,
            pool_type="max",
            pool_padding="VALID",
            data_format="NHWC")

        # test SAME
        out_7 = fluid.layers.pool2d(
            input=input_NCHW,
            pool_size=[4, 4],
            pool_type="avg",
            pool_padding="SAME",
            data_format="NCHW")

        out_8 = fluid.layers.pool2d(
            input=input_NHWC,
            pool_size=[4, 4],
            pool_type="max",
            pool_padding="SAME",
            data_format="NHWC")

        # test negetive
        out_9 = fluid.layers.pool2d(
            input=input_NHWC_negetive,
            pool_size=ksize,
            pool_type="avg",
            pool_padding=[0, 0],
            data_format="NHWC")
        assert out_9.shape == (2, -1, 3, 3)

        out_10 = fluid.layers.pool2d(
            input=input_NCHW_negetive,
            pool_size=ksize,
            pool_type="avg",
            pool_padding=[0, 0],
            data_format="NCHW")
        assert out_10.shape == (2, 3, -1, -1)

        exe = fluid.Executor(place=fluid.MLUPlace(0))
        [res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8] = exe.run(
            fluid.default_main_program(),
            feed={
                "input_NHWC": x_NHWC,
                "input_NCHW": x_NCHW,
                "input_NHWC_negetive": x_NHWC,
                "input_NCHW_negetive": x_NCHW
            },
            fetch_list=[
                out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8
            ])

        assert np.allclose(
            res_1,
            pool2D_forward_naive(
                x=x_NHWC,
                ksize=ksize,
                pool_type="max",
                strides=[1, 1],
                paddings=[1, 1],
                data_format="NHWC"))

        assert np.allclose(
            res_2,
            pool2D_forward_naive(
                x=x_NHWC,
                ksize=ksize,
                pool_type="avg",
                strides=[1, 1],
                paddings=[1, 1, 1, 1],
                data_format="NHWC"))
        assert np.allclose(
            res_3,
            pool2D_forward_naive(
                x=x_NCHW,
                ksize=ksize,
                pool_type="avg",
                strides=[1, 1],
                paddings=[1, 1, 1, 1],
                data_format="NCHW"),
            rtol=0.07,
            atol=1e-05)

        assert np.allclose(
            res_4,
            pool2D_forward_naive(
                x=x_NCHW,
                ksize=ksize,
                pool_type="avg",
                strides=[1, 1],
                paddings=[1, 2, 1, 0],
                data_format="NCHW"),
            rtol=0.07,
            atol=1e-05)

        # VALID
        assert np.allclose(
            res_5,
            pool2D_forward_naive(
                x=x_NCHW,
                ksize=ksize,
                pool_type="avg",
                strides=[1, 1],
                paddings=[10, 20],  # any ele is ok
                padding_algorithm="VALID",
                data_format="NCHW"),
            rtol=0.07,
            atol=1e-05)
        assert np.allclose(
            res_6,
            pool2D_forward_naive(
                x=x_NHWC,
                ksize=ksize,
                pool_type="max",
                strides=[1, 1],
                paddings=[10, 20],
                padding_algorithm="VALID",
                data_format="NHWC"))
        # SAME
        assert np.allclose(
            res_7,
            pool2D_forward_naive(
                x=x_NCHW,
                ksize=[4, 4],
                pool_type="avg",
                strides=[1, 1],
                paddings=[10, 20],
                padding_algorithm="SAME",
                data_format="NCHW"),
            rtol=0.07,
            atol=1e-05)

        assert np.allclose(
            res_8,
            pool2D_forward_naive(
                x=x_NHWC,
                ksize=[4, 4],
                pool_type="max",
                strides=[1, 1],
                paddings=[10, 20],
                padding_algorithm="SAME",
                data_format="NHWC"))


class TestPool2DAPI_Error(unittest.TestCase):
    def test_api(self):
        input_NHWC = fluid.layers.data(
            name="input_NHWC",
            shape=[2, 5, 5, 3],
            append_batch_size=False,
            dtype="float32")
        ksize = [3, 3]

        # data_format value error
        def run_2():
            out_2 = fluid.layers.pool2d(
                input=input_NHWC,
                pool_size=ksize,
                pool_type="max",
                pool_padding=[1, 1],
                data_format="NHWCC")

        self.assertRaises(ValueError, run_2)

        # padding str value error
        def run_3():
            out_3 = fluid.layers.pool2d(
                input=input_NHWC,
                pool_size=ksize,
                pool_type="max",
                pool_padding="VALIDSAME",
                data_format="NHWC")

        self.assertRaises(ValueError, run_3)

        # padding str valid and ceil_mode value error
        def run_4():
            out_4 = fluid.layers.pool2d(
                input=input_NHWC,
                pool_size=ksize,
                pool_type="max",
                pool_padding="VALID",
                ceil_mode=True,
                data_format="NHWC")

        self.assertRaises(ValueError, run_4)

        # padding with 8 ele. value error
        def run_5():
            out_5 = fluid.layers.pool2d(
                input=input_NHWC,
                pool_size=ksize,
                pool_type="max",
                pool_padding=[[1, 1], [0, 0], [0, 0], [1, 1]],
                data_format="NHWC")

        self.assertRaises(ValueError, run_5)


class TestDygraphPool2DAPIError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # the input of Pool2D must be Variable.
            data1 = np.random.random((3, 32, 32, 5)).astype('float32')
            pool2d = fluid.dygraph.Pool2D(
                pool_size=2,
                pool_type='max',
                pool_stride=1,
                global_pooling=False)
            self.assertRaises(TypeError, pool2d, data1)

            # the input dtype of mlu Pool2D must be float16 or float32 
            data2 = fluid.layers.data(
                name='x1', shape=[3, 32, 32, 5], dtype="int32")
            self.assertRaises(TypeError, pool2d, data2)

    def test_data_format_error(self):
        with program_guard(Program(), Program()):
            # the data_format must be 'NCHW' or 'NHWC'
            data1 = np.random.random((3, 32, 32, 5)).astype('float32')
            self.assertRaises(
                ValueError,
                fluid.dygraph.Pool2D,
                pool_size=2,
                pool_type='max',
                pool_stride=1,
                global_pooling=False,
                data_format='NWHC')


class TestDygraphPool2DAPI(unittest.TestCase):
    def test_nhwc(self):
        with fluid.dygraph.guard():
            data = np.random.random((3, 32, 32, 5)).astype('float32')
            x = fluid.dygraph.to_variable(data)
            pool2d = fluid.dygraph.Pool2D(
                pool_size=2,
                pool_type='max',
                pool_stride=1,
                pool_padding=[0, 0],
                global_pooling=False,
                data_format='NHWC')
            out1 = pool2d(x)
            out2 = pool2D_forward_naive(
                data, [2, 2], [1, 1],
                paddings=[0, 0],
                pool_type='max',
                data_format='NHWC')
            self.assertTrue(np.allclose(out1.numpy(), out2))

    def test_lower_case(self):
        with fluid.dygraph.guard():
            data = np.random.random((3, 32, 32, 5)).astype('float32')
            x = fluid.dygraph.to_variable(data)
            pool2d = fluid.dygraph.Pool2D(
                pool_size=2,
                pool_type='max',
                pool_stride=1,
                pool_padding=[0, 0],
                global_pooling=False,
                data_format='nhwc')
            out1 = pool2d(x)
            out2 = pool2D_forward_naive(
                data, [2, 2], [1, 1],
                paddings=[0, 0],
                pool_type='max',
                data_format='NHWC')
            self.assertTrue(np.allclose(out1.numpy(), out2))

    def test_upper_case(self):
        with fluid.dygraph.guard():
            data = np.random.random((3, 32, 32, 5)).astype('float32')
            x = fluid.dygraph.to_variable(data)
            pool2d = fluid.dygraph.Pool2D(
                pool_size=2,
                pool_type='MAX',
                pool_stride=1,
                pool_padding=[0, 0],
                global_pooling=False,
                data_format='nhwc')
            out1 = pool2d(x)
            out2 = pool2D_forward_naive(
                data, [2, 2], [1, 1],
                paddings=[0, 0],
                pool_type='max',
                data_format='NHWC')
            self.assertTrue(np.allclose(out1.numpy(), out2))


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

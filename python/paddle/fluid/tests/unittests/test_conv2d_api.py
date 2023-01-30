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

<<<<<<< HEAD
import unittest

=======
from __future__ import print_function

import unittest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import numpy as np

import paddle

paddle.enable_static()
<<<<<<< HEAD
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestConv2DAPI(unittest.TestCase):
    def test_api(self):

        input_NHWC = paddle.static.data(
            name="input_NHWC",
            shape=[2, 5, 5, 3],
            dtype="float32",
        )

        input_NCHW = paddle.static.data(
            name="input_NCHW",
            shape=[2, 3, 5, 5],
            dtype="float32",
        )

        paddle.static.nn.conv2d(
            input=input_NHWC,
            num_filters=3,
            filter_size=[3, 3],
            stride=[1, 1],
            padding=0,
            dilation=[1, 1],
            groups=1,
            data_format="NCHW",
        )

        paddle.static.nn.conv2d(
            input=input_NCHW,
            num_filters=3,
            filter_size=[3, 3],
            stride=[1, 1],
            padding=[1, 2, 1, 0],
            dilation=[1, 1],
            groups=1,
            data_format="NCHW",
        )

        paddle.static.nn.conv2d(
            input=input_NCHW,
            num_filters=3,
            filter_size=[3, 3],
            stride=[1, 1],
            padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
            dilation=[1, 1],
            groups=1,
            data_format="NCHW",
        )

        paddle.static.nn.conv2d(
            input=input_NHWC,
            num_filters=3,
            filter_size=[3, 3],
            stride=[1, 1],
            padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
            dilation=[1, 1],
            groups=1,
            data_format="NHWC",
        )

        paddle.static.nn.conv2d(
            input=input_NCHW,
            num_filters=3,
            filter_size=[3, 3],
            stride=[1, 1],
            padding="SAME",
            dilation=[1, 1],
            groups=1,
            data_format="NCHW",
        )

        paddle.static.nn.conv2d(
            input=input_NCHW,
            num_filters=3,
            filter_size=[3, 3],
            stride=[1, 1],
            padding="VALID",
            dilation=[1, 1],
            groups=1,
            data_format="NCHW",
        )

    def test_depthwise_conv2d(self):
        x_var = paddle.uniform((2, 8, 8, 4), dtype='float32', min=-1.0, max=1.0)
        conv = paddle.nn.Conv2D(
            in_channels=4,
            out_channels=4,
            kernel_size=(3, 3),
            groups=4,
            data_format='NHWC',
        )
=======
import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test import OpTest
from paddle.fluid import Program, program_guard


class TestConv2DAPI(unittest.TestCase):

    def test_api(self):

        input_NHWC = fluid.layers.data(name="input_NHWC",
                                       shape=[2, 5, 5, 3],
                                       append_batch_size=False,
                                       dtype="float32")

        input_NCHW = fluid.layers.data(name="input_NCHW",
                                       shape=[2, 3, 5, 5],
                                       append_batch_size=False,
                                       dtype="float32")

        fluid.layers.conv2d(input=input_NHWC,
                            num_filters=3,
                            filter_size=[3, 3],
                            stride=[1, 1],
                            padding=0,
                            dilation=[1, 1],
                            groups=1,
                            data_format="NCHW")

        fluid.layers.conv2d(input=input_NCHW,
                            num_filters=3,
                            filter_size=[3, 3],
                            stride=[1, 1],
                            padding=[1, 2, 1, 0],
                            dilation=[1, 1],
                            groups=1,
                            data_format="NCHW")

        fluid.layers.conv2d(input=input_NCHW,
                            num_filters=3,
                            filter_size=[3, 3],
                            stride=[1, 1],
                            padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                            dilation=[1, 1],
                            groups=1,
                            data_format="NCHW")

        fluid.layers.conv2d(input=input_NHWC,
                            num_filters=3,
                            filter_size=[3, 3],
                            stride=[1, 1],
                            padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
                            dilation=[1, 1],
                            groups=1,
                            data_format="NHWC")

        fluid.layers.conv2d(input=input_NCHW,
                            num_filters=3,
                            filter_size=[3, 3],
                            stride=[1, 1],
                            padding="SAME",
                            dilation=[1, 1],
                            groups=1,
                            data_format="NCHW")

        fluid.layers.conv2d(input=input_NCHW,
                            num_filters=3,
                            filter_size=[3, 3],
                            stride=[1, 1],
                            padding="VALID",
                            dilation=[1, 1],
                            groups=1,
                            data_format="NCHW")

    def test_depthwise_conv2d(self):
        x_var = paddle.uniform((2, 8, 8, 4), dtype='float32', min=-1., max=1.)
        conv = paddle.nn.Conv2D(in_channels=4,
                                out_channels=4,
                                kernel_size=(3, 3),
                                groups=4,
                                data_format='NHWC')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        y_var = conv(x_var)


class TestConv2DAPI_Error(unittest.TestCase):
<<<<<<< HEAD
    def test_api(self):
        input = paddle.static.data(
            name="input",
            shape=[2, 5, 5, 5],
            dtype="float32",
        )

        # ValueError: cudnn
        def run_1():
            paddle.static.nn.conv2d(
                input=input,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding=0,
                dilation=[1, 1],
                groups=1,
                use_cudnn=[0],
                data_format="NCHW",
            )
=======

    def test_api(self):
        input = fluid.layers.data(name="input",
                                  shape=[2, 5, 5, 5],
                                  append_batch_size=False,
                                  dtype="float32")

        # ValueError: cudnn
        def run_1():
            fluid.layers.conv2d(input=input,
                                num_filters=3,
                                filter_size=[3, 3],
                                stride=[1, 1],
                                padding=0,
                                dilation=[1, 1],
                                groups=1,
                                use_cudnn=[0],
                                data_format="NCHW")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(ValueError, run_1)

        # ValueError: data_format
        def run_2():
<<<<<<< HEAD
            paddle.static.nn.conv2d(
                input=input,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding=0,
                dilation=[1, 1],
                groups=1,
                use_cudnn=False,
                data_format="NCHWC",
            )
=======
            fluid.layers.conv2d(input=input,
                                num_filters=3,
                                filter_size=[3, 3],
                                stride=[1, 1],
                                padding=0,
                                dilation=[1, 1],
                                groups=1,
                                use_cudnn=False,
                                data_format="NCHWC")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(ValueError, run_2)

        # ValueError: padding
        def run_3():
<<<<<<< HEAD
            paddle.static.nn.conv2d(
                input=input,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding="SAMEE",
                dilation=[1, 1],
                groups=1,
                use_cudnn=False,
                data_format="NCHW",
            )
=======
            fluid.layers.conv2d(input=input,
                                num_filters=3,
                                filter_size=[3, 3],
                                stride=[1, 1],
                                padding="SAMEE",
                                dilation=[1, 1],
                                groups=1,
                                use_cudnn=False,
                                data_format="NCHW")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(ValueError, run_3)

        def run_4():
<<<<<<< HEAD
            paddle.static.nn.conv2d(
                input=input,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding=[[0, 1], [0, 1], [0, 1], [0, 1]],
                dilation=[1, 1],
                groups=1,
                use_cudnn=False,
                data_format="NCHW",
            )
=======
            fluid.layers.conv2d(input=input,
                                num_filters=3,
                                filter_size=[3, 3],
                                stride=[1, 1],
                                padding=[[0, 1], [0, 1], [0, 1], [0, 1]],
                                dilation=[1, 1],
                                groups=1,
                                use_cudnn=False,
                                data_format="NCHW")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(ValueError, run_4)

        def run_5():
<<<<<<< HEAD
            paddle.static.nn.conv2d(
                input=input,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding=[[0, 1], [0, 1], [0, 1], [0, 1]],
                dilation=[1, 1],
                groups=1,
                use_cudnn=False,
                data_format="NHWC",
            )
=======
            fluid.layers.conv2d(input=input,
                                num_filters=3,
                                filter_size=[3, 3],
                                stride=[1, 1],
                                padding=[[0, 1], [0, 1], [0, 1], [0, 1]],
                                dilation=[1, 1],
                                groups=1,
                                use_cudnn=False,
                                data_format="NHWC")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(ValueError, run_5)

        # ValueError: channel dimmention
<<<<<<< HEAD
        x = paddle.static.data(
            name="x",
            shape=[2, 5, 5, -1],
            dtype="float32",
        )

        def run_6():
            paddle.static.nn.conv2d(
                input=x,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding=0,
                dilation=[1, 1],
                groups=1,
                use_cudnn=False,
                data_format="NHWC",
            )
=======
        x = fluid.layers.data(name="x",
                              shape=[2, 5, 5, -1],
                              append_batch_size=False,
                              dtype="float32")

        def run_6():
            fluid.layers.conv2d(input=x,
                                num_filters=3,
                                filter_size=[3, 3],
                                stride=[1, 1],
                                padding=0,
                                dilation=[1, 1],
                                groups=1,
                                use_cudnn=False,
                                data_format="NHWC")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(ValueError, run_6)

        # ValueError: groups
        def run_7():
<<<<<<< HEAD
            paddle.static.nn.conv2d(
                input=input,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding=0,
                dilation=[1, 1],
                groups=3,
                use_cudnn=False,
                data_format="NHWC",
            )
=======
            fluid.layers.conv2d(input=input,
                                num_filters=3,
                                filter_size=[3, 3],
                                stride=[1, 1],
                                padding=0,
                                dilation=[1, 1],
                                groups=3,
                                use_cudnn=False,
                                data_format="NHWC")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(ValueError, run_7)

        # ValueError: filter num
        def run_8():
<<<<<<< HEAD
            paddle.static.nn.conv2d(
                input=input,
                num_filters=0,
                filter_size=0,
                stride=0,
                padding=0,
                dilation=0,
                groups=1,
                use_cudnn=False,
                data_format="NCHW",
            )
=======
            fluid.layers.conv2d(input=input,
                                num_filters=0,
                                filter_size=0,
                                stride=0,
                                padding=0,
                                dilation=0,
                                groups=1,
                                use_cudnn=False,
                                data_format="NCHW")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(ValueError, run_8)

        # ValueError: groups
        def run_9():
<<<<<<< HEAD
            paddle.static.nn.conv2d(
                input=input,
                num_filters=0,
                filter_size=0,
                stride=0,
                padding=0,
                dilation=0,
                groups=0,
                use_cudnn=False,
                data_format="NCHW",
            )
=======
            fluid.layers.conv2d(input=input,
                                num_filters=0,
                                filter_size=0,
                                stride=0,
                                padding=0,
                                dilation=0,
                                groups=0,
                                use_cudnn=False,
                                data_format="NCHW")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(ValueError, run_9)

        # ValueError: stride
        def run_10():
<<<<<<< HEAD
            paddle.static.nn.conv2d(
                input=input,
                num_filters=1,
                filter_size=1,
                stride=0,
                padding=0,
                dilation=0,
                groups=1,
                use_cudnn=False,
                data_format="NCHW",
            )
=======
            fluid.layers.conv2d(input=input,
                                num_filters=1,
                                filter_size=1,
                                stride=0,
                                padding=0,
                                dilation=0,
                                groups=1,
                                use_cudnn=False,
                                data_format="NCHW")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(ValueError, run_10)

    def test_api_with_error_input(self):
<<<<<<< HEAD
        input = paddle.static.data(
            name="error_input",
            shape=[1],
            dtype="float32",
        )

        # ValueError: cudnn
        def run_1():
            paddle.static.nn.conv2d(
                input=input,
                num_filters=0,
                filter_size=0,
                stride=0,
                padding=0,
                dilation=0,
                groups=0,
                use_cudnn=False,
                data_format="NCHW",
            )
=======
        input = fluid.layers.data(name="error_input",
                                  shape=[1],
                                  append_batch_size=False,
                                  dtype="float32")

        # ValueError: cudnn
        def run_1():
            fluid.layers.conv2d(input=input,
                                num_filters=0,
                                filter_size=0,
                                stride=0,
                                padding=0,
                                dilation=0,
                                groups=0,
                                use_cudnn=False,
                                data_format="NCHW")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(ValueError, run_1)


# --------- test environment variable ------
@unittest.skipIf(
    not (core.is_compiled_with_cuda() or core.is_compiled_with_rocm()),
<<<<<<< HEAD
    "core is not compiled with CUDA or ROCM",
)
class TestConv2DEnviron(unittest.TestCase):
    def run1(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            inputs = paddle.static.data(
                shape=[2, 3, 5, 5],
                name="inputs",
                dtype="float32",
            )
            result = paddle.static.nn.conv2d(
                input=inputs,
                num_filters=4,
                filter_size=[3, 3],
                stride=[1, 1],
                padding=0,
                dilation=[1, 1],
                groups=1,
                data_format="NCHW",
            )
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            fetches = exe.run(
                fluid.default_main_program(),
                feed={"inputs": self.input_np},
                fetch_list=[result],
            )
=======
    "core is not compiled with CUDA or ROCM")
class TestConv2DEnviron(unittest.TestCase):

    def run1(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            inputs = fluid.layers.data(shape=[2, 3, 5, 5],
                                       append_batch_size=False,
                                       name="inputs",
                                       dtype="float32")
            result = fluid.layers.conv2d(input=inputs,
                                         num_filters=4,
                                         filter_size=[3, 3],
                                         stride=[1, 1],
                                         padding=0,
                                         dilation=[1, 1],
                                         groups=1,
                                         data_format="NCHW")
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            fetches = exe.run(fluid.default_main_program(),
                              feed={"inputs": self.input_np},
                              fetch_list=[result])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def run2(self, place):
        with fluid.dygraph.guard(place):
            inputs = fluid.dygraph.to_variable(self.input_np)
<<<<<<< HEAD
            conv = paddle.nn.Conv2D(
                in_channels=3,
                out_channels=4,
                kernel_size=(3, 3),
                data_format="NCHW",
=======
            conv = paddle.nn.Conv2D(in_channels=3,
                                    out_channels=4,
                                    kernel_size=(3, 3),
                                    data_format="NCHW")
            result = conv(inputs)

    def run3(self, place):
        with fluid.dygraph.guard(place):
            inputs = fluid.dygraph.to_variable(self.input_np)
            conv = paddle.fluid.dygraph.nn.Conv2D(
                num_channels=3,
                num_filters=4,
                filter_size=(3, 3),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            )
            result = conv(inputs)

    def run_all(self, place):
        self.run1(place)
        self.run2(place)
<<<<<<< HEAD
=======
        self.run3(place)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_environ(self):
        self.input_np = np.random.random([2, 3, 5, 5]).astype("float32")
        for place in [paddle.CPUPlace(), paddle.CUDAPlace(0)]:
            fluid.set_flags({'FLAGS_conv2d_disable_cudnn': False})
            self.run_all(place)
            fluid.set_flags({'FLAGS_conv2d_disable_cudnn': True})
            self.run_all(place)


if __name__ == '__main__':
    unittest.main()

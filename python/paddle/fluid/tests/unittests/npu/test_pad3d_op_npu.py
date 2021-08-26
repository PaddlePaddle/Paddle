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

import unittest
import numpy as np
import sys
sys.path.append("..")
import op_test
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard, Executor, default_main_program
import paddle.fluid as fluid


class TestPad3dNPUOp(op_test.OpTest):
    def setUp(self):
        paddle.enable_static()
        self.__class__.use_npu = True
        self.op_type = "pad3d"
        self.place = paddle.NPUPlace(0)

        self.x_type = "float32"
        self.mode = "constant"
        self.variable_paddings = False
        self.initTestCase()

        self.value = 0  #Asend npu only support constant_values = 0 right now.
        self.inputs = {'X': np.random.random(self.shape).astype(self.x_type)}
        self.attrs = {}
        if self.variable_paddings:
            self.attrs['paddings'] = []
            self.inputs['Paddings'] = np.array(self.paddings).flatten().astype(
                "int32")
        else:
            self.attrs['paddings'] = np.array(self.paddings).flatten().astype(
                "int32")
        self.attrs['value'] = self.value
        self.attrs['mode'] = self.mode
        self.attrs['data_format'] = self.data_format
        if self.data_format == "NCDHW":
            paddings = [
                (0, 0),
                (0, 0),
                (self.paddings[4], self.paddings[5]),
                (self.paddings[2], self.paddings[3]),
                (self.paddings[0], self.paddings[1]),
            ]
        else:
            paddings = [
                (0, 0),
                (self.paddings[4], self.paddings[5]),
                (self.paddings[2], self.paddings[3]),
                (self.paddings[0], self.paddings[1]),
                (0, 0),
            ]

        out = np.pad(self.inputs['X'],
                     paddings,
                     mode=self.mode,
                     constant_values=self.value)

        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 0, 0, 0, 0, 0]
        self.data_format = "NCDHW"


class TestCase1(TestPad3dNPUOp):
    def initTestCase(self):
        self.shape = (3, 4, 5, 6, 7)
        self.paddings = [0, 1, 2, 3, 4, 5]
        self.data_format = "NCDHW"
        self.x_type = "float16"

    def test_check_grad(self):
        self.__class__.no_need_check_grad = True
        pass


class TestCase2(TestPad3dNPUOp):
    def initTestCase(self):
        self.shape = (4, 5, 6, 7, 8)
        self.paddings = [1, 1, 1, 1, 1, 1]
        self.data_format = "NDHWC"
        self.variable_paddings = True


class TestPadAPI(unittest.TestCase):
    def _get_numpy_out(self,
                       input_data,
                       pad,
                       mode,
                       value=0,
                       data_format="NCDHW"):
        if mode == "constant" and len(pad) == len(input_data.shape) * 2:
            pad = np.reshape(pad, (-1, 2)).tolist()
        elif data_format == "NCDHW":
            pad = [
                (0, 0),
                (0, 0),
                (pad[4], pad[5]),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
            ]
        elif data_format == "NDHWC":
            pad = [
                (0, 0),
                (pad[4], pad[5]),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
                (0, 0),
            ]
        elif data_format == "NCHW":
            pad = [
                (0, 0),
                (0, 0),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
            ]
        elif data_format == "NHWC":
            pad = [
                (0, 0),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
                (0, 0),
            ]
        elif data_format == "NCL":
            pad = [
                (0, 0),
                (0, 0),
                (pad[0], pad[1]),
            ]
        elif data_format == "NLC":
            pad = [
                (0, 0),
                (pad[0], pad[1]),
                (0, 0),
            ]

        out = np.pad(input_data, pad, mode=mode, constant_values=value)
        return out

    def test_static(self):
        paddle.enable_static()
        self.place = fluid.NPUPlace(0) if fluid.core.is_compiled_with_npu(
        ) else fluid.CPUPlace()
        with program_guard(Program(), Program()):
            input_shape = (1, 2, 3, 4, 5)
            pad = [1, 2, 1, 1, 3, 4]
            mode = "constant"
            value = 0
            input_data = np.random.rand(*input_shape).astype(np.float32)
            x = paddle.fluid.data(name="x", shape=input_shape)
            result1 = F.pad(x=x,
                            pad=pad,
                            value=value,
                            mode=mode,
                            data_format="NCDHW")
            result2 = F.pad(x=x,
                            pad=pad,
                            value=value,
                            mode=mode,
                            data_format="NDHWC")
            exe = Executor(self.place)
            fetches = exe.run(default_main_program(),
                              feed={"x": input_data},
                              fetch_list=[result1, result2])

            np_out1 = self._get_numpy_out(
                input_data, pad, mode, value, data_format="NCDHW")
            np_out2 = self._get_numpy_out(
                input_data, pad, mode, value, data_format="NDHWC")
            self.assertTrue(np.allclose(fetches[0], np_out1))
            self.assertTrue(np.allclose(fetches[1], np_out2))

    def test_dygraph_1(self):
        paddle.disable_static()
        paddle.device.set_device("npu")
        input_shape = (1, 2, 3, 4, 5)
        pad = [1, 2, 1, 1, 3, 4]

        mode = "constant"
        value = 0
        input_data = np.random.rand(*input_shape).astype(np.float32)
        tensor_data = paddle.to_tensor(input_data)

        np_out1 = self._get_numpy_out(
            input_data, pad, mode, value, data_format="NCDHW")
        np_out2 = self._get_numpy_out(
            input_data, pad, mode, value, data_format="NDHWC")

        y1 = F.pad(tensor_data,
                   pad=pad,
                   mode=mode,
                   value=value,
                   data_format="NCDHW")
        y2 = F.pad(tensor_data,
                   pad=pad,
                   mode=mode,
                   value=value,
                   data_format="NDHWC")

        self.assertTrue(np.allclose(y1.numpy(), np_out1))
        self.assertTrue(np.allclose(y2.numpy(), np_out2))

    def test_dygraph_2(self):
        paddle.disable_static()
        paddle.device.set_device("npu")
        input_shape = (2, 3, 4, 5)
        pad = [1, 1, 3, 4]

        mode = "constant"
        value = 0
        input_data = np.random.rand(*input_shape).astype(np.float32)
        tensor_data = paddle.to_tensor(input_data)

        np_out1 = self._get_numpy_out(
            input_data, pad, mode, value, data_format="NCHW")
        np_out2 = self._get_numpy_out(
            input_data, pad, mode, value, data_format="NHWC")

        y1 = F.pad(tensor_data,
                   pad=pad,
                   mode=mode,
                   value=value,
                   data_format="NCHW")
        y2 = F.pad(tensor_data,
                   pad=pad,
                   mode=mode,
                   value=value,
                   data_format="NHWC")

        self.assertTrue(np.allclose(y1.numpy(), np_out1))
        self.assertTrue(np.allclose(y2.numpy(), np_out2))

    def test_dygraph_3(self):
        paddle.disable_static()
        paddle.device.set_device("npu")
        input_shape = (3, 4, 5)
        pad = [3, 4]

        mode = "constant"
        value = 0
        input_data = np.random.rand(*input_shape).astype(np.float32)
        tensor_data = paddle.to_tensor(input_data)

        np_out1 = self._get_numpy_out(
            input_data, pad, mode, value, data_format="NCL")
        np_out2 = self._get_numpy_out(
            input_data, pad, mode, value, data_format="NLC")

        y1 = F.pad(tensor_data,
                   pad=pad,
                   mode=mode,
                   value=value,
                   data_format="NCL")
        y2 = F.pad(tensor_data,
                   pad=pad,
                   mode=mode,
                   value=value,
                   data_format="NLC")

        self.assertTrue(np.allclose(y1.numpy(), np_out1))
        self.assertTrue(np.allclose(y2.numpy(), np_out2))


class TestPad1dAPI(unittest.TestCase):
    def _get_numpy_out(self,
                       input_data,
                       pad,
                       mode,
                       value=0.0,
                       data_format="NCL"):
        if data_format == "NCL":
            pad = [
                (0, 0),
                (0, 0),
                (pad[0], pad[1]),
            ]
        else:
            pad = [
                (0, 0),
                (pad[0], pad[1]),
                (0, 0),
            ]

        out = np.pad(input_data, pad, mode=mode, constant_values=value)
        return out

    def test_class(self):
        paddle.disable_static()
        paddle.device.set_device("npu")
        input_shape = (3, 4, 5)
        pad = [1, 2]
        pad_int = 1
        value = 0
        input_data = np.random.rand(*input_shape).astype(np.float32)

        pad_constant = nn.Pad1D(padding=pad, mode="constant", value=value)
        pad_constant_int = nn.Pad1D(
            padding=pad_int, mode="constant", value=value)

        data = paddle.to_tensor(input_data)

        output = pad_constant(data)
        np_out = self._get_numpy_out(
            input_data, pad, "constant", value=value, data_format="NCL")
        self.assertTrue(np.allclose(output.numpy(), np_out))

        output = pad_constant_int(data)
        np_out = self._get_numpy_out(
            input_data, [pad_int] * 2,
            "constant",
            value=value,
            data_format="NCL")
        self.assertTrue(np.allclose(output.numpy(), np_out))


class TestPad2dAPI(unittest.TestCase):
    def _get_numpy_out(self,
                       input_data,
                       pad,
                       mode,
                       value=0.0,
                       data_format="NCHW"):
        if data_format == "NCHW":
            pad = [
                (0, 0),
                (0, 0),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
            ]
        else:
            pad = [
                (0, 0),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
                (0, 0),
            ]

        out = np.pad(input_data, pad, mode=mode, constant_values=value)
        return out

    def test_class(self):
        paddle.disable_static()
        paddle.device.set_device("npu")
        input_shape = (3, 4, 5, 6)
        pad = [1, 2, 2, 1]
        pad_int = 1
        value = 0
        input_data = np.random.rand(*input_shape).astype(np.float32)

        pad_constant = nn.Pad2D(padding=pad, mode="constant", value=value)
        pad_constant_int = nn.Pad2D(
            padding=pad_int, mode="constant", value=value)

        data = paddle.to_tensor(input_data)

        output = pad_constant(data)
        np_out = self._get_numpy_out(
            input_data, pad, "constant", value=value, data_format="NCHW")
        self.assertTrue(np.allclose(output.numpy(), np_out))

        output = pad_constant_int(data)
        np_out = self._get_numpy_out(
            input_data, [pad_int] * 4,
            "constant",
            value=value,
            data_format="NCHW")
        self.assertTrue(np.allclose(output.numpy(), np_out))


class TestPad3dAPI(unittest.TestCase):
    def _get_numpy_out(self,
                       input_data,
                       pad,
                       mode,
                       value=0.0,
                       data_format="NCDHW"):
        if data_format == "NCDHW":
            pad = [
                (0, 0),
                (0, 0),
                (pad[4], pad[5]),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
            ]
        else:
            pad = [
                (0, 0),
                (pad[4], pad[5]),
                (pad[2], pad[3]),
                (pad[0], pad[1]),
                (0, 0),
            ]

        out = np.pad(input_data, pad, mode=mode, constant_values=value)
        return out

    def test_class(self):
        paddle.disable_static()
        paddle.device.set_device("npu")
        input_shape = (3, 4, 5, 6, 7)
        pad = [1, 2, 2, 1, 1, 0]
        pad_int = 1
        value = 0
        input_data = np.random.rand(*input_shape).astype(np.float32)

        pad_constant = nn.Pad3D(padding=pad, mode="constant", value=value)
        pad_constant_int = nn.Pad3D(
            padding=pad_int, mode="constant", value=value)

        data = paddle.to_tensor(input_data)

        output = pad_constant(data)
        np_out = self._get_numpy_out(
            input_data, pad, "constant", value=value, data_format="NCDHW")
        self.assertTrue(np.allclose(output.numpy(), np_out))

        output = pad_constant_int(data)
        np_out = self._get_numpy_out(
            input_data, [pad_int] * 6,
            "constant",
            value=value,
            data_format="NCDHW")
        self.assertTrue(np.allclose(output.numpy(), np_out))


class TestPad3dOpNpuError(unittest.TestCase):
    def test_errors(self):
        def test_value():
            input_shape = (1, 2, 3, 4, 5)
            data = np.random.rand(*input_shape).astype(np.float32)
            x = paddle.fluid.data(name="x", shape=input_shape)
            y = F.pad(x, pad=[1, 1, 1, 1, 1, 1], value=1, mode='constant')
            place = paddle.NPUPlace()
            exe = Executor(place)
            outputs = exe.run(feed={'x': data}, fetch_list=[y.name])

        def test_mode_1():
            input_shape = (1, 2, 3, 4, 5)
            data = np.random.rand(*input_shape).astype(np.float32)
            x = paddle.fluid.data(name="x", shape=input_shape)
            y = F.pad(x, pad=[1, 1, 1, 1, 1, 1], mode='reflect')
            place = paddle.NPUPlace()
            exe = Executor(place)
            outputs = exe.run(feed={'x': data}, fetch_list=[y.name])

        def test_mode_2():
            input_shape = (1, 2, 3, 4, 5)
            data = np.random.rand(*input_shape).astype(np.float32)
            x = paddle.fluid.data(name="x", shape=input_shape)
            y = F.pad(x, pad=[1, 1, 1, 1, 1, 1], mode='replicate')
            place = paddle.NPUPlace()
            exe = Executor(place)
            outputs = exe.run(feed={'x': data}, fetch_list=[y.name])

        def test_mode_3():
            input_shape = (1, 2, 3, 4, 5)
            data = np.random.rand(*input_shape).astype(np.float32)
            x = paddle.fluid.data(name="x", shape=input_shape)
            y = F.pad(x, pad=[1, 1, 1, 1, 1, 1], mode='circular')
            place = paddle.CPUPlace()
            exe = Executor(place)
            outputs = exe.run(feed={'x': data}, fetch_list=[y.name])

        self.assertRaises(Exception, test_value)

        self.assertRaises(Exception, test_mode_1)

        self.assertRaises(Exception, test_mode_2)

        self.assertRaises(Exception, test_mode_3)


class TestPadDataformatError(unittest.TestCase):
    def test_errors(self):
        def test_ncl():
            input_shape = (1, 2, 3, 4)
            pad = paddle.to_tensor(np.array([2, 1, 2, 1]).astype('int32'))
            data = np.arange(
                np.prod(input_shape), dtype=np.float64).reshape(input_shape) + 1
            my_pad = nn.Pad1D(padding=pad, mode="replicate", data_format="NCL")
            data = paddle.to_tensor(data)
            result = my_pad(data)

        def test_nchw():
            input_shape = (1, 2, 4)
            pad = paddle.to_tensor(np.array([2, 1, 2, 1]).astype('int32'))
            data = np.arange(
                np.prod(input_shape), dtype=np.float64).reshape(input_shape) + 1
            my_pad = nn.Pad1D(padding=pad, mode="replicate", data_format="NCHW")
            data = paddle.to_tensor(data)
            result = my_pad(data)

        def test_ncdhw():
            input_shape = (1, 2, 3, 4)
            pad = paddle.to_tensor(np.array([2, 1, 2, 1]).astype('int32'))
            data = np.arange(
                np.prod(input_shape), dtype=np.float64).reshape(input_shape) + 1
            my_pad = nn.Pad1D(
                padding=pad, mode="replicate", data_format="NCDHW")
            data = paddle.to_tensor(data)
            result = my_pad(data)

        self.assertRaises(AssertionError, test_ncl)

        self.assertRaises(AssertionError, test_nchw)

        self.assertRaises(AssertionError, test_ncdhw)


if __name__ == '__main__':
    unittest.main()

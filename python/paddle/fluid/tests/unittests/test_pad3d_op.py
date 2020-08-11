#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid.dygraph as dg
import paddle.fluid.core as core

from paddle.fluid import Program, program_guard, Executor, default_main_program


class TestPad3dOp(OpTest):
    def setUp(self):
        self.value = 0.0
        self.variable_paddings = False
        self.initTestCase()
        self.op_type = "pad3d"
        self.inputs = {'X': np.random.random(self.shape).astype("float64")}
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
        if self.mode == "constant":
            out = np.pad(self.inputs['X'],
                         paddings,
                         mode=self.mode,
                         constant_values=self.value)
        elif self.mode == "reflect":
            out = np.pad(self.inputs['X'], paddings, mode=self.mode)
        elif self.mode == "replicate":
            out = np.pad(self.inputs['X'], paddings, mode="edge")
        elif self.mode == "circular":
            out = np.pad(self.inputs['X'], paddings, mode="wrap")
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')

    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 0, 0, 0, 0, 0]
        self.mode = "constant"
        self.data_format = "NCDHW"
        self.pad_value = 0.0


class TestCase1(TestPad3dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 4, 5]
        self.mode = "constant"
        self.data_format = "NCDHW"
        self.value = 1.0


class TestCase2(TestPad3dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [1, 1, 1, 1, 1, 1]
        self.mode = "constant"
        self.data_format = "NDHWC"
        self.value = 1.0


class TestCase3(TestPad3dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 1, 0, 2, 3]
        self.mode = "reflect"
        self.data_format = "NCDHW"


class TestCase4(TestPad3dOp):
    def initTestCase(self):
        self.shape = (4, 4, 4, 4, 4)
        self.paddings = [0, 1, 2, 1, 2, 3]
        self.mode = "reflect"
        self.data_format = "NDHWC"


class TestCase5(TestPad3dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 2, 1]
        self.mode = "replicate"
        self.data_format = "NCDHW"


class TestCase6(TestPad3dOp):
    def initTestCase(self):
        self.shape = (4, 4, 4, 4, 4)
        self.paddings = [5, 4, 2, 1, 2, 3]
        self.mode = "replicate"
        self.data_format = "NDHWC"


class TestCase7(TestPad3dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 2, 1]
        self.mode = "circular"
        self.data_format = "NCDHW"


class TestCase8(TestPad3dOp):
    def initTestCase(self):
        self.shape = (4, 4, 4, 4, 4)
        self.paddings = [0, 1, 2, 1, 2, 3]
        self.mode = "circular"
        self.data_format = "NDHWC"


class TestCase9(TestPad3dOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 3, 1]
        self.mode = "reflect"
        self.data_format = "NCDHW"
        self.variable_paddings = True


class TestPad3dDygraph(unittest.TestCase):
    def _get_numpy_out(self, input_data, pad, mode, value, data_format="NCDHW"):
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

        if mode == "constant":
            out = np.pad(input_data, pad, mode=mode, constant_values=value)
        elif mode == "reflect":
            out = np.pad(input_data, pad, mode=mode)
        elif mode == "replicate":
            out = np.pad(input_data, pad, mode="edge")
        elif mode == "circular":
            out = np.pad(input_data, pad, mode="wrap")

        return out

    def test_dygraph(self):

        input_shape = (1, 2, 3, 4, 5)
        pad = [1, 2, 1, 1, 3, 4]
        mode = "constant"
        value = 100
        input_data = np.random.rand(*input_shape).astype(np.float32)
        np_out = self._get_numpy_out(input_data, pad, mode, value)
        place = paddle.CPUPlace()
        with dg.guard(place) as g:
            input = dg.to_variable(input_data)
            output = F.pad(input=input, pad=pad, mode=mode, value=value)
            self.assertTrue(np.allclose(output.numpy(), np_out))


class TestPadAPI(unittest.TestCase):
    def _get_numpy_out(self, input_data, pad, mode, value, data_format="NCDHW"):
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

        if mode == "constant":
            out = np.pad(input_data, pad, mode=mode, constant_values=value)
        elif mode == "reflect":
            out = np.pad(input_data, pad, mode=mode)
        elif mode == "replicate":
            out = np.pad(input_data, pad, mode="edge")
        elif mode == "circular":
            out = np.pad(input_data, pad, mode="wrap")

        return out

    def setUp(self):
        self.places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        with program_guard(Program(), Program()):
            input_shape = (1, 2, 3, 4, 5)
            pad = [1, 2, 1, 1, 3, 4]
            mode = "constant"
            value = 100
            input_data = np.random.rand(*input_shape).astype(np.float32)
            x = paddle.data(name="x", shape=input_shape)
            result = F.pad(input=x, pad=pad, value=value, mode='constant')
            exe = Executor(place)
            fetches = exe.run(default_main_program(),
                              feed={"x": input_data},
                              fetch_list=[result])

            np_out = self._get_numpy_out(input_data, pad, mode, value)

            self.assertTrue(np.allclose(fetches[0], np_out))

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            input_shape = (1, 2, 3, 4, 5)
            pad = [1, 2, 1, 1, 3, 4]
            mode = "constant"
            value = 100
            input_data = np.random.rand(*input_shape).astype(np.float32)
            np_out = self._get_numpy_out(input_data, pad, mode, value)
            with dg.guard(place) as g:
                input = dg.to_variable(input_data)
                output = F.pad(input=input, pad=pad, mode=mode, value=value)
                self.assertTrue(np.allclose(output.numpy(), np_out))


class TestPad1dClass(unittest.TestCase):
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

        if mode == "constant":
            out = np.pad(input_data, pad, mode=mode, constant_values=value)
        elif mode == "reflect":
            out = np.pad(input_data, pad, mode=mode)
        elif mode == "replicate":
            out = np.pad(input_data, pad, mode="edge")

        return out

    def setUp(self):
        self.places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_class(self):
        for place in self.places:
            input_shape = (3, 4, 5)
            pad = [1, 2]
            value = 100
            input_data = np.random.rand(*input_shape).astype(np.float32)

            pad_reflection = nn.ReflectionPad1d(padding=pad)
            pad_replication = nn.ReplicationPad1d(padding=pad)
            pad_constant = nn.ConstantPad1d(padding=pad, value=value)

            with dg.guard(place) as g:
                data = paddle.fluid.dygraph.to_variable(input_data)

                output = pad_reflection(data)
                np_out = self._get_numpy_out(
                    input_data, pad, "reflect", data_format="NCL")
                self.assertTrue(np.allclose(output.numpy(), np_out))

                output = pad_replication(data)
                np_out = self._get_numpy_out(
                    input_data, pad, "replicate", data_format="NCL")
                self.assertTrue(np.allclose(output.numpy(), np_out))

                output = pad_constant(data)
                np_out = self._get_numpy_out(
                    input_data, pad, "constant", value=value, data_format="NCL")
                self.assertTrue(np.allclose(output.numpy(), np_out))


class TestPad3dOpError(unittest.TestCase):
    def test_errors(self):
        def test_variable():
            input_shape = (1, 2, 3, 4, 5)
            data = np.random.rand(*input_shape).astype(np.float32)
            F.pad(input=data, paddings=[1, 1, 1, 1, 1, 1])

        def test_reflect_1():
            input_shape = (1, 2, 3, 4, 5)
            data = np.random.rand(*input_shape).astype(np.float32)
            x = paddle.data(name="x", shape=input_shape)
            y = F.pad(x, pad=[5, 6, 1, 1, 1, 1], value=1, mode='reflect')
            place = paddle.CPUPlace()
            exe = Executor(place)
            outputs = exe.run(feed={'x': data}, fetch_list=[y.name])

        def test_reflect_2():
            input_shape = (1, 2, 3, 4, 5)
            data = np.random.rand(*input_shape).astype(np.float32)
            x = paddle.data(name="x", shape=input_shape)
            y = F.pad(x, pad=[1, 1, 4, 3, 1, 1], value=1, mode='reflect')
            place = paddle.CPUPlace()
            exe = Executor(place)
            outputs = exe.run(feed={'x': data}, fetch_list=[y.name])

        def test_reflect_3():
            input_shape = (1, 2, 3, 4, 5)
            data = np.random.rand(*input_shape).astype(np.float32)
            x = paddle.data(name="x", shape=input_shape)
            y = F.pad(x, pad=[1, 1, 1, 1, 2, 3], value=1, mode='reflect')
            place = paddle.CPUPlace()
            exe = Executor(place)
            outputs = exe.run(feed={'x': data}, fetch_list=[y.name])

        self.assertRaises(TypeError, test_variable)

        self.assertRaises(Exception, test_reflect_1)

        self.assertRaises(Exception, test_reflect_2)

        self.assertRaises(Exception, test_reflect_3)


if __name__ == '__main__':
    unittest.main()

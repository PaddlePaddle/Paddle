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
from op_test import get_device_place, is_custom_device
from utils import dygraph_guard, static_guard

import paddle
from paddle import base
from paddle.base import Program, program_guard


class TestChunkOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The type of axis in chunk_op should be int or Variable.
            def test_axis_type():
                x1 = paddle.static.data(shape=[4], dtype='float16', name='x3')
                paddle.chunk(x=x1, chunks=2, axis=3.2)

            self.assertRaises(TypeError, test_axis_type)

            # The type of num_or_sections in chunk_op should be int, tuple or list.
            def test_chunks_type():
                x4 = paddle.static.data(shape=[4], dtype='float16', name='x4')
                paddle.chunk(x=x4, chunks=2.1, axis=3)

            self.assertRaises(TypeError, test_chunks_type)

            def test_axis_type_tensor():
                x5 = paddle.static.data(shape=[4], dtype='float16', name='x6')
                paddle.chunk(x=x5, chunks=2, axis=3.2)

            self.assertRaises(TypeError, test_axis_type_tensor)

        with paddle.base.dygraph.guard():

            def test_0_chunks_tensor():
                x = paddle.uniform([1, 1, 1], dtype='float32')
                paddle.chunk(x, chunks=0)

            self.assertRaises(ValueError, test_0_chunks_tensor)

            def test_negative_chunks_tensor():
                x = paddle.uniform([2, 3, 4], dtype='float32')
                paddle.chunk(x, chunks=-1)

            self.assertRaises(ValueError, test_negative_chunks_tensor)

            def test_chunks_greater_than_dim():
                x = paddle.uniform([2, 3, 4], dtype='float32')
                # axis=1, shape=3, chunks=5 > 3
                paddle.chunk(x, chunks=5, axis=1)

            self.assertRaises(ValueError, test_chunks_greater_than_dim)


class API_TestChunk(unittest.TestCase):
    def test_out(self):
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data(
                'data1', shape=[4, 6, 6], dtype='float64'
            )
            data2 = paddle.static.data('data2', shape=[1], dtype='int32')
            x0, x1, x2 = paddle.chunk(data1, chunks=3, axis=data2)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            input1 = np.random.random([4, 6, 6]).astype('float64')
            input2 = np.array([2]).astype('int32')
            (
                r0,
                r1,
                r2,
            ) = exe.run(
                feed={"data1": input1, "data2": input2}, fetch_list=[x0, x1, x2]
            )
            ex_x0, ex_x1, ex_x2 = np.array_split(input1, 3, axis=2)
            np.testing.assert_allclose(ex_x0, r0, rtol=1e-05)
            np.testing.assert_allclose(ex_x1, r1, rtol=1e-05)
            np.testing.assert_allclose(ex_x2, r2, rtol=1e-05)


class API_TestChunk1(unittest.TestCase):
    def test_out(self):
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data(
                'data1', shape=[4, 6, 6], dtype='float64'
            )
            x0, x1, x2 = paddle.chunk(data1, chunks=3, axis=2)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            input1 = np.random.random([4, 6, 6]).astype('float64')
            (
                r0,
                r1,
                r2,
            ) = exe.run(feed={"data1": input1}, fetch_list=[x0, x1, x2])
            ex_x0, ex_x1, ex_x2 = np.array_split(input1, 3, axis=2)
            np.testing.assert_allclose(ex_x0, r0, rtol=1e-05)
            np.testing.assert_allclose(ex_x1, r1, rtol=1e-05)
            np.testing.assert_allclose(ex_x2, r2, rtol=1e-05)


class API_TestChunkZeroSize1(unittest.TestCase):
    def test_out(self):
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data(
                'data1', shape=[0, 1, 1, 4], dtype='float32'
            )
            x0, x1, x2, x3 = paddle.chunk(data1, chunks=4, axis=-1)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            input1 = np.random.random([0, 1, 1, 4]).astype('float32')
            (
                r0,
                r1,
                r2,
                r3,
            ) = exe.run(feed={"data1": input1}, fetch_list=[x0, x1, x2, x3])
            ex_x0, ex_x1, ex_x2, ex_x3 = np.array_split(input1, 4, axis=-1)
            np.testing.assert_allclose(ex_x0, r0, rtol=1e-05)
            np.testing.assert_allclose(ex_x1, r1, rtol=1e-05)
            np.testing.assert_allclose(ex_x2, r2, rtol=1e-05)
            np.testing.assert_allclose(ex_x3, r3, rtol=1e-05)


class API_TestDygraphChunk(unittest.TestCase):
    def test_out1(self):
        with base.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            x0, x1, x2 = paddle.chunk(input, chunks=3, axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.array_split(input_1, 3, axis=1)
        np.testing.assert_allclose(ex_x0, x0_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x1, x1_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x2, x2_out, rtol=1e-05)

    def test_out2(self):
        with base.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("bool")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            x0, x1, x2 = paddle.chunk(input, chunks=3, axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.array_split(input_1, 3, axis=1)
        np.testing.assert_allclose(ex_x0, x0_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x1, x1_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x2, x2_out, rtol=1e-05)

    def test_axis_tensor_input(self):
        with base.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            num1 = paddle.full(shape=[1], fill_value=1, dtype='int32')
            x0, x1, x2 = paddle.chunk(input, chunks=3, axis=num1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.array_split(input_1, 3, axis=1)
        np.testing.assert_allclose(ex_x0, x0_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x1, x1_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x2, x2_out, rtol=1e-05)


class TestChunkCompatibility(unittest.TestCase):
    def setUp(self):
        self.places = [paddle.CPUPlace()]
        if paddle.base.core.is_compiled_with_cuda() or is_custom_device():
            self.places.append(get_device_place())
        self.func = paddle.chunk
        self.init_data()
        self.init_case()

    def init_data(self):
        self.shape = [6, 4]
        self.dtype = 'float32'
        self.np_input = np.random.random(self.shape).astype(self.dtype)
        self.chunks = 2
        self.axis = 0
        self.np_out = np.array_split(self.np_input, self.chunks, axis=self.axis)

    def init_case(self):
        params = [
            ['x', 'input'],  # param1
            ['chunks'],  # param2
            ['axis', 'dim'],  # param3
        ]

        # Generate all valid combinations
        def generate_cases(param_groups, case_list):
            from itertools import product

            for combo in product(*[[None, *names] for names in param_groups]):
                args = ['pos' if p is None else 'kw' for p in combo]
                if args == sorted(args, key=lambda x: x != 'pos'):
                    case_list.append(combo)

        # paddle.chunk()
        self.test_cases = []
        generate_cases(params, self.test_cases)
        # x.chunk()
        self.tensor_test_cases = []
        generate_cases(params[1:], self.tensor_test_cases)

    def _build_args_kwargs(self, param_names, params):
        args = []
        kwargs = {}
        for name, param in zip(param_names, params):
            if name is None:
                args.append(param)
            else:
                kwargs[name] = param
        return args, kwargs

    def test_dygraph_compatibility(self):
        with dygraph_guard():
            for place in self.places:
                paddle.device.set_device(place)
                x = paddle.to_tensor(self.np_input)
                # paddle.
                for param_names in self.test_cases:
                    args, kwargs = self._build_args_kwargs(
                        param_names, (x, self.chunks, self.axis)
                    )
                    outs = self.func(*args, **kwargs)
                    for out, np_out in zip(outs, self.np_out):
                        np.testing.assert_allclose(
                            np_out, out.numpy(), rtol=1e-10
                        )
                # paddle.Tensor.
                for param_names in self.tensor_test_cases:
                    args, kwargs = self._build_args_kwargs(
                        param_names, (self.chunks, self.axis)
                    )
                    outs = x.chunk(*args, **kwargs)
                    for out, np_out in zip(outs, self.np_out):
                        np.testing.assert_allclose(
                            np_out, out.numpy(), rtol=1e-10
                        )

    def test_static_compatibility(self):
        with static_guard():
            for place in self.places:
                main = paddle.static.Program()
                startup = paddle.static.Program()
                with base.program_guard(main, startup):
                    x = paddle.static.data(
                        name="x", shape=self.shape, dtype=self.dtype
                    )
                    # paddle.
                    for param_names in self.test_cases:
                        args, kwargs = self._build_args_kwargs(
                            param_names, (x, self.chunks, self.axis)
                        )

                        outs = self.func(*args, **kwargs)

                        exe = base.Executor(place)
                        fetches = exe.run(
                            main,
                            feed={"x": self.np_input},
                            fetch_list=outs,
                        )
                        for fetch, np_out in zip(fetches, self.np_out):
                            np.testing.assert_allclose(
                                np_out, fetch, rtol=1e-10
                            )
                    # paddle.Tensor.
                    for param_names in self.tensor_test_cases:
                        args, kwargs = self._build_args_kwargs(
                            param_names, (self.chunks, self.axis)
                        )
                        outs = x.chunk(*args, **kwargs)

                        exe = base.Executor(place)
                        fetches = exe.run(
                            main,
                            feed={"x": self.np_input},
                            fetch_list=outs,
                        )
                        for fetch, np_out in zip(fetches, self.np_out):
                            np.testing.assert_allclose(
                                np_out, fetch, rtol=1e-10
                            )


if __name__ == '__main__':
    unittest.main()

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

import os
import sys
import tempfile
import unittest

sys.path.append("../../legacy_test")
import numpy as np
from op_test import OpTest

import paddle
import paddle.inference as paddle_infer
from paddle import base
from paddle.base.framework import in_dygraph_mode, in_pir_mode

paddle.enable_static()


class TestBincountOpAPI(unittest.TestCase):
    """Test bincount api."""

    def test_static_graph(self):
        startup_program = paddle.static.Program()
        train_program = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup_program):
            inputs = paddle.static.data(name='input', dtype='int64', shape=[7])
            weights = paddle.static.data(
                name='weights', dtype='int64', shape=[7]
            )
            output = paddle.bincount(inputs, weights=weights)
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            exe.run(startup_program)
            img = np.array([0, 1, 1, 3, 2, 1, 7]).astype(np.int64)
            w = np.array([0, 1, 1, 2, 2, 1, 0]).astype(np.int64)
            res = exe.run(
                train_program,
                feed={'input': img, 'weights': w},
                fetch_list=[output],
            )
            actual = np.array(res[0])
            expected = np.bincount(img, weights=w)
            self.assertTrue(
                (actual == expected).all(),
                msg='bincount output is wrong, out =' + str(actual),
            )

    def test_dygraph(self):
        with base.dygraph.guard():
            inputs_np = np.array([0, 1, 1, 3, 2, 1, 7]).astype(np.int64)
            inputs = paddle.to_tensor(inputs_np)
            actual = paddle.bincount(inputs)
            expected = np.bincount(inputs)
            self.assertTrue(
                (actual.numpy() == expected).all(),
                msg='bincount output is wrong, out =' + str(actual.numpy()),
            )


class TestBincountOpError(unittest.TestCase):
    """Test bincount op error."""

    def run_network(self, net_func):
        with base.dygraph.guard():
            net_func()

    def test_input_value_error(self):
        """Test input tensor should be non-negative."""

        def net_func():
            input_value = paddle.to_tensor([1, 2, 3, 4, -5])
            paddle.bincount(input_value)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_input_shape_error(self):
        """Test input tensor should be 1-D tansor."""

        def net_func():
            input_value = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
            paddle.bincount(input_value)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_minlength_value_error(self):
        """Test minlength is non-negative ints."""

        def net_func():
            input_value = paddle.to_tensor([1, 2, 3, 4, 5])
            paddle.bincount(input_value, minlength=-1)

        with base.dygraph.guard():
            if in_dygraph_mode():
                # InvalidArgument for phi BincountKernel
                with self.assertRaises(ValueError):
                    self.run_network(net_func)
            else:
                # OutOfRange for EqualGreaterThanChecker
                with self.assertRaises(IndexError):
                    self.run_network(net_func)

    def test_input_type_errors(self):
        """Test input tensor should only contain non-negative ints."""

        def net_func():
            input_value = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            paddle.bincount(input_value)

        with self.assertRaises(TypeError):
            self.run_network(net_func)

    def test_weights_shape_error(self):
        """Test weights tensor should have the same shape as input tensor."""

        def net_func():
            input_value = paddle.to_tensor([1, 2, 3, 4, 5])
            weights = paddle.to_tensor([1, 1, 1, 1, 1, 1])
            paddle.bincount(input_value, weights=weights)

        with self.assertRaises(ValueError):
            self.run_network(net_func)


class TestBincountOp(OpTest):
    # without weights
    def setUp(self):
        self.op_type = "bincount"
        self.python_api = paddle.bincount
        self.init_test_case()
        self.inputs = {"X": self.np_input}
        self.attrs = {"minlength": self.minlength}
        self.outputs = {"Out": self.Out}

    def init_test_case(self):
        self.minlength = 0
        self.np_input = np.random.randint(low=0, high=20, size=10)
        self.Out = np.bincount(self.np_input, minlength=self.minlength)

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)


class TestCase1(TestBincountOp):
    # with weights(FLOAT32)
    def setUp(self):
        self.op_type = "bincount"
        self.python_api = paddle.bincount
        self.init_test_case()
        self.inputs = {"X": self.np_input, "Weights": self.np_weights}
        self.attrs = {"minlength": self.minlength}
        self.outputs = {"Out": self.Out}

    def init_test_case(self):
        self.minlength = 0
        self.np_weights = np.random.randint(low=0, high=20, size=10).astype(
            np.float32
        )
        self.np_input = np.random.randint(low=0, high=20, size=10)
        self.Out = np.bincount(
            self.np_input, weights=self.np_weights, minlength=self.minlength
        ).astype(np.float32)


class TestCase2(TestBincountOp):
    # with weights(other)
    def setUp(self):
        self.op_type = "bincount"
        self.python_api = paddle.bincount
        self.init_test_case()
        self.inputs = {"X": self.np_input, "Weights": self.np_weights}
        self.attrs = {"minlength": self.minlength}
        self.outputs = {"Out": self.Out}

    def init_test_case(self):
        self.minlength = 0
        self.np_weights = np.random.randint(low=0, high=20, size=10)
        self.np_input = np.random.randint(low=0, high=20, size=10)
        self.Out = np.bincount(
            self.np_input, weights=self.np_weights, minlength=self.minlength
        )


class TestCase3(TestBincountOp):
    # empty input
    def init_test_case(self):
        self.minlength = 0
        self.np_input = np.array([], dtype=np.int64)
        self.Out = np.bincount(self.np_input, minlength=self.minlength)


class TestCase4(TestBincountOp):
    # with input(INT32)
    def init_test_case(self):
        self.minlength = 0
        self.np_input = np.random.randint(low=0, high=20, size=10).astype(
            np.int32
        )
        self.Out = np.bincount(self.np_input, minlength=self.minlength)


class TestCase5(TestBincountOp):
    # with minlength greater than max(X)
    def init_test_case(self):
        self.minlength = 20
        self.np_input = np.random.randint(low=0, high=10, size=10)
        self.Out = np.bincount(self.np_input, minlength=self.minlength)


class TestCase6(TestBincountOp):
    # with bigger input size
    def init_test_case(self):
        self.minlength = 0
        self.np_input = np.random.randint(low=0, high=10, size=1024)
        self.Out = np.bincount(self.np_input, minlength=self.minlength)


class TestTensorMinlength(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        paddle.seed(2022)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_path = os.path.join(
            self.temp_dir.name, 'tensor_minlength_bincount'
        )
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_dygraph(self):
        paddle.disable_static()
        x = np.random.randint(0, 10, [20])
        minlength = 2
        np_out = np.bincount(x, minlength=minlength)
        pd_out = paddle.bincount(
            paddle.to_tensor(x), minlength=paddle.to_tensor([2], dtype='int32')
        )
        np.testing.assert_allclose(np_out, pd_out.numpy())

    def test_static_and_infer(self):
        paddle.enable_static()
        np_x = np.random.randn(100).astype('float32')
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            # run static
            x = paddle.static.data(shape=np_x.shape, name='x', dtype=np_x.dtype)
            linear = paddle.nn.Linear(np_x.shape[0], np_x.shape[0])
            linear_out = linear(x)
            relu_out = paddle.nn.functional.relu(linear_out)
            minlength = paddle.full([1], 3, dtype='int32')
            out = paddle.bincount(
                paddle.cast(relu_out, 'int32'), minlength=minlength
            )

            exe = paddle.static.Executor(self.place)
            exe.run(startup_prog)
            static_out = exe.run(feed={'x': np_x}, fetch_list=[out])

            # run infer
            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            if in_pir_mode():
                config = paddle_infer.Config(
                    self.save_path + '.json', self.save_path + '.pdiparams'
                )
                config.enable_new_executor()
                config.enable_new_ir()
            else:
                config = paddle_infer.Config(
                    self.save_path + '.pdmodel', self.save_path + '.pdiparams'
                )

            if paddle.is_compiled_with_cuda():
                config.enable_use_gpu(100, 0)
            else:
                config.disable_gpu()

            predictor = paddle_infer.create_predictor(config)
            input_names = predictor.get_input_names()
            input_handle = predictor.get_input_handle(input_names[0])
            fake_input = np_x
            input_handle.reshape(np_x.shape)
            input_handle.copy_from_cpu(fake_input)
            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])
            infer_out = output_handle.copy_to_cpu()
            np.testing.assert_allclose(static_out[0], infer_out)


if __name__ == "__main__":
    unittest.main()

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
import tempfile
import numpy as np
from op_test import OpTest, skip_check_grad_ci, check_out_dtype
import paddle
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard
import paddle.inference as paddle_infer


class ApiMinTest(unittest.TestCase):

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

    def test_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data = paddle.static.data("data", shape=[10, 10], dtype="float32")
            result_min = paddle.min(x=data, axis=1)
            exe = paddle.static.Executor(self.place)
            input_data = np.random.rand(10, 10).astype(np.float32)
            res, = exe.run(feed={"data": input_data}, fetch_list=[result_min])
        self.assertEqual((res == np.min(input_data, axis=1)).all(), True)

        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data = paddle.static.data("data", shape=[10, 10], dtype="int64")
            result_min = paddle.min(x=data, axis=0)
            exe = paddle.static.Executor(self.place)
            input_data = np.random.randint(10, size=(10, 10)).astype(np.int64)
            res, = exe.run(feed={"data": input_data}, fetch_list=[result_min])
        self.assertEqual((res == np.min(input_data, axis=0)).all(), True)

        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data = paddle.static.data("data", shape=[10, 10], dtype="int64")
            result_min = paddle.min(x=data, axis=(0, 1))
            exe = paddle.static.Executor(self.place)
            input_data = np.random.randint(10, size=(10, 10)).astype(np.int64)
            res, = exe.run(feed={"data": input_data}, fetch_list=[result_min])
        self.assertEqual((res == np.min(input_data, axis=(0, 1))).all(), True)

    def test_errors(self):
        paddle.enable_static()

        def test_input_type():
            with paddle.static.program_guard(paddle.static.Program(),
                                             paddle.static.Program()):
                data = np.random.rand(10, 10)
                result_min = paddle.min(x=data, axis=0)

        self.assertRaises(TypeError, test_input_type)

    def test_imperative_api(self):
        paddle.disable_static()
        np_x = np.array([10, 10]).astype('float64')
        x = paddle.to_tensor(np_x)
        z = paddle.min(x, axis=0)
        np_z = z.numpy()
        z_expected = np.array(np.min(np_x, axis=0))
        self.assertEqual((np_z == z_expected).all(), True)

    def test_eager_api(self):
        with _test_eager_guard():
            self.test_imperative_api()


class TestOutDtype(unittest.TestCase):

    def test_min(self):
        api_fn = paddle.min
        shape = [10, 16]
        check_out_dtype(api_fn,
                        in_specs=[(shape, )],
                        expect_dtypes=['float32', 'float64', 'int32', 'int64'])


class TestMinWithTensorAxis1(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_path = os.path.join(self.temp_dir.name,
                                      'min_with_tensor_axis')
        self.place = paddle.CUDAPlace(
            0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
        self.init_data()

    def tearDwon(self):
        self.temp_dir.cleanup()

    def init_data(self):
        self.x = paddle.randn([2, 3, 4, 5, 5], dtype='float32')
        self.axis = paddle.to_tensor([1, 2], dtype='int64')

    def test_dygraph(self):
        self.x.stop_gradient = False
        pd_out = paddle.min(self.x, self.axis)
        np_out = np.min(self.x.numpy(), tuple(self.axis.numpy()))

        self.assertTrue(
            np.array_equal(pd_out.numpy() if pd_out.size > 1 else pd_out.item(),
                           np_out))
        pd_out.backward()
        self.assertEqual(self.x.gradient().shape, tuple(self.x.shape))

    def test_static_and_infer(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        starup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, starup_prog):
            # run static
            x = paddle.static.data(shape=self.x.shape,
                                   name='x',
                                   dtype=self.x.dtype)
            axis = paddle.static.data(shape=self.axis.shape,
                                      name='axis',
                                      dtype=self.axis.dtype)
            fc = paddle.nn.Linear(self.x.shape[-1], 6)
            fc_out = fc(x)
            out = paddle.min(fc_out, axis)
            exe = paddle.static.Executor(self.place)
            exe.run(starup_prog)
            static_out = exe.run(feed={
                'x': self.x.numpy(),
                'axis': self.axis.numpy()
            },
                                 fetch_list=[out])

            # run infer
            paddle.static.save_inference_model(self.save_path, [x, axis], [out],
                                               exe)
            config = paddle_infer.Config(self.save_path + '.pdmodel',
                                         self.save_path + '.pdiparams')
            predictor = paddle_infer.create_predictor(config)
            input_names = predictor.get_input_names()
            input_handle = predictor.get_input_handle(input_names[0])
            fake_input = self.x.numpy()
            input_handle.reshape(self.x.shape)
            input_handle.copy_from_cpu(fake_input)
            input_handle = predictor.get_input_handle(input_names[1])
            fake_input = self.axis.numpy()
            input_handle.reshape(self.axis.shape)
            input_handle.copy_from_cpu(fake_input)
            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])
            infer_out = output_handle.copy_to_cpu()

            self.assertTrue(np.array_equal(static_out[0], infer_out))
            paddle.disable_static()


class TestMinWithTensorAxis2(TestMinWithTensorAxis1):

    def init_data(self):
        self.x = paddle.randn([3, 4, 7], dtype='float32')
        self.axis = paddle.to_tensor([0, 1, 2], dtype='int64')


if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle

paddle.enable_static()
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import PaddleDType
from paddle.inference import Config, create_predictor
from paddle.inference import get_trt_compile_version, get_trt_runtime_version


class TestInferenceApi(unittest.TestCase):
    def test_inference_api(self):
        tensor32 = np.random.randint(10, 20, size=[20, 2]).astype('int32')
        paddletensor32 = PaddleTensor(tensor32)
        dtype32 = paddletensor32.dtype
        self.assertEqual(dtype32, PaddleDType.INT32)
        self.assertEqual(
            paddletensor32.data.tolist('int32'), tensor32.ravel().tolist()
        )
        paddletensor32.data.reset(tensor32)
        self.assertEqual(
            paddletensor32.as_ndarray().ravel().tolist(),
            tensor32.ravel().tolist(),
        )

        tensor64 = np.random.randint(10, 20, size=[20, 2]).astype('int64')
        paddletensor64 = PaddleTensor(tensor64)
        dtype64 = paddletensor64.dtype
        self.assertEqual(dtype64, PaddleDType.INT64)
        self.assertEqual(
            paddletensor64.data.tolist('int64'), tensor64.ravel().tolist()
        )
        paddletensor64.data.reset(tensor64)
        self.assertEqual(
            paddletensor64.as_ndarray().ravel().tolist(),
            tensor64.ravel().tolist(),
        )

        tensor_float = np.random.randn(20, 2).astype('float32')
        paddletensor_float = PaddleTensor(tensor_float)
        dtype_float = paddletensor_float.dtype
        self.assertEqual(dtype_float, PaddleDType.FLOAT32)
        self.assertEqual(
            paddletensor_float.data.tolist('float32'),
            tensor_float.ravel().tolist(),
        )
        paddletensor_float.data.reset(tensor_float)
        self.assertEqual(
            paddletensor_float.as_ndarray().ravel().tolist(),
            tensor_float.ravel().tolist(),
        )


def get_sample_model():
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        data = fluid.data(name="data", shape=[-1, 6, 64, 64], dtype="float32")
        conv_out = fluid.layers.conv2d(
            input=data,
            num_filters=3,
            filter_size=3,
            groups=1,
            padding=0,
            bias_attr=False,
            act=None,
        )
    exe.run(startup_program)
    serialized_program = paddle.static.serialize_program(
        data, conv_out, program=main_program
    )
    serialized_params = paddle.static.serialize_persistables(
        data, conv_out, executor=exe, program=main_program
    )
    return serialized_program, serialized_params


class TestInferenceBaseAPI(unittest.TestCase):
    def get_config(self, model, params):
        config = Config()
        config.set_model_buffer(model, len(model), params, len(params))
        config.enable_use_gpu(100, 0)
        return config

    def test_apis(self):
        print('trt compile version:', get_trt_compile_version())
        print('trt runtime version:', get_trt_runtime_version())
        program, params = get_sample_model()
        config = self.get_config(program, params)
        predictor = create_predictor(config)
        in_names = predictor.get_input_names()
        in_handle = predictor.get_input_handle(in_names[0])
        in_data = np.ones((1, 6, 32, 32)).astype(np.float32)
        in_handle.copy_from_cpu(in_data)
        predictor.run()

    def test_wrong_input(self):
        with self.assertRaises(TypeError):
            program, params = get_sample_model()
            config = self.get_config(program, params)
            predictor = create_predictor(config)
            in_names = predictor.get_input_names()
            in_handle = predictor.get_input_handle(in_names[0])
            in_data = np.ones((1, 6, 64, 64)).astype(np.float32)
            in_handle.copy_from_cpu(list(in_data))
            predictor.run()


if __name__ == '__main__':
    unittest.main()

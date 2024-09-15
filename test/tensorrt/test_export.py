# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import numpy as np

import paddle
import paddle.inference as paddle_infer
import paddle.nn.functional as F
from paddle import nn
from paddle.static import InputSpec
from paddle.tensorrt.export import (
    Input,
    TensorRTConfig,
    convert,
    convert_loaded_model,
)
from paddle.tensorrt.util import (
    predict_program,
)


class CumsumModel(nn.Layer):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        linear_out = self.linear(x)
        relu_out = F.relu(linear_out)
        axis = paddle.full([1], 2, dtype='int64')
        out = paddle.cumsum(relu_out, axis=axis)
        return out


class TestConvertLoadedModel(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_path = os.path.join(self.temp_dir.name, 'tensor_axis_cumsum')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_paddle_to_tensorrt_conversion_cumsum(self):
        paddle.enable_static()
        np_x = np.random.randn(9, 10, 11).astype('float32')

        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    shape=np_x.shape, name='x', dtype=np_x.dtype
                )
                model = CumsumModel(input_dim=np_x.shape[-1])
                out = model(x)
                loss = paddle.mean(out)
                sgd = paddle.optimizer.SGD(learning_rate=0.0)
                sgd.minimize(paddle.mean(out))

                exe = paddle.static.Executor(self.place)
                exe.run(startup_prog)
                static_out = exe.run(feed={'x': np_x}, fetch_list=[out])

                # run infer
                paddle.static.save_inference_model(
                    self.save_path, [x], [out], exe
                )

                config = paddle_infer.Config(
                    self.save_path + '.json', self.save_path + '.pdiparams'
                )
                config.enable_new_ir()
                config.enable_new_executor()
                config.use_optimized_model(True)

            # Set input
            input_config = Input(
                min_input_shape=(9, 10, 11),
                optim_input_shape=(9, 10, 11),
                max_input_shape=(9, 10, 11),
            )
            # Create a TensorRTConfig with inputs as a required field.
            trt_config = TensorRTConfig(inputs=[input_config])

            trt_save_path = os.path.join(self.temp_dir.name, 'trt')
            trt_config.save_model_dir = trt_save_path

            model_dir = self.save_path
            # Obtain tensorrt_engine_op by passing the model path and trt_config.(converted_program)
            program_with_trt = convert_loaded_model(model_dir, trt_config)

            # Create a config for inference.
            config = paddle_infer.Config(
                trt_config.save_model_dir + '.json',
                trt_config.save_model_dir + '.pdiparams',
            )

            if paddle.is_compiled_with_cuda():
                config.enable_use_gpu(100, 0)
            else:
                config.disable_gpu()
            predictor = paddle_infer.create_predictor(config)
            input_names = predictor.get_input_names()

        paddle.disable_static()
        for i, input_instrance in enumerate(trt_config.inputs):
            min_data, _, max_data = input_instrance[i].generate_input_data()
            model_inputs = paddle.to_tensor(min_data)
            output_converted = predictor.run([model_inputs])


class TestConvert(unittest.TestCase):
    def test_run(self):
        with paddle.pir_utils.IrGuard():
            input_config = Input(
                min_input_shape=(9, 10, 11),
                optim_input_shape=(9, 10, 11),
                max_input_shape=(9, 10, 11),
            )
            trt_config = TensorRTConfig(inputs=[input_config])
            for i, input_instrance in enumerate(trt_config.inputs):
                min_data, _, max_data = input_instrance[i].generate_input_data()
                paddle.disable_static()
                x = paddle.to_tensor(min_data)
                net = CumsumModel(input_dim=min_data.shape[-1])
                out = net(x)

                input_spec = [InputSpec(shape=min_data.shape, dtype='float32')]
                program_with_trt, scope = convert(
                    net,
                    input_spec=input_spec,
                    config=trt_config,
                    full_graph=True,
                )
                output_var = program_with_trt.list_vars()[-1]
                output_converted = predict_program(
                    program_with_trt,
                    {"x": min_data},
                    [output_var],
                    scope=scope,
                )
                output_expected = out.numpy()
                output_converted_np = output_converted[0]

                # Check that the results are close to each other within a tolerance of 1e-2
                np.testing.assert_allclose(
                    output_expected,
                    output_converted_np,
                    rtol=1e-2,
                    atol=1e-2,
                    err_msg="Outputs are not within the 1e-2 tolerance",
                )


if __name__ == "__main__":
    unittest.main()

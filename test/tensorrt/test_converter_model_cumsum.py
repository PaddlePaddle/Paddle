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
from paddle.tensorrt.export import (
    Input,
    TensorRTConfig,
    export_loaded_model,
)


class TestConverterCumsumOp(unittest.TestCase):
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
                linear = paddle.nn.Linear(np_x.shape[-1], np_x.shape[-1])
                linear_out = linear(x)
                relu_out = paddle.nn.functional.relu(linear_out)
                axis = paddle.full([1], 2, dtype='int64')
                out = paddle.cumsum(relu_out, axis=axis)
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
            trt_config.disable_ops = "pd_op.matmul"

            model_dir = self.save_path
            # Obtain tensorrt_engine_op by passing the model path and trt_config.(converted_program)
            program_with_trt = export_loaded_model(model_dir, trt_config)

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


if __name__ == "__main__":
    unittest.main()

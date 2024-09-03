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
from paddle.framework import use_pir_api
from paddle.tensorrt.export import (
    Input,
    TensorRTConfig,
    get_trt_program,
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
        # Step1: get program and init fake inputs
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

                if use_pir_api():
                    config = paddle_infer.Config(
                        self.save_path + '.json', self.save_path + '.pdiparams'
                    )
                    config.enable_new_ir()
                    config.enable_new_executor()
                    config.use_optimized_model(True)
                else:
                    config = paddle_infer.Config(
                        self.save_path + '.pdmodel',
                        self.save_path + '.pdiparams',
                    )

        with paddle.pir_utils.IrGuard():
            input_config = Input(
                min_input_shape=(9, 10, 11),
                optim_input_shape=(9, 10, 11),
                max_input_shape=(9, 10, 11),
            )
            trt_config = TensorRTConfig()
            trt_config.inputs = [input_config]
            trt_config.save_model_dir = self.temp_dir.name
            trt_config.save_model_prefix = 'trt'
            program_with_trt, trt_save_path = get_trt_program(
                self.temp_dir.name, "tensor_axis_cumsum", trt_config, True
            )

            config = paddle_infer.Config(
                trt_save_path + '.json', trt_save_path + '.pdiparams'
            )

            if paddle.is_compiled_with_cuda():
                config.enable_use_gpu(100, 0)
            else:
                config.disable_gpu()
            predictor = paddle_infer.create_predictor(config)
            input_names = predictor.get_input_names()

            input_handle = predictor.get_input_handle(input_names[0])
            for input_instrance in trt_config.inputs:
                min_data, _, max_data = input_instrance.generate_input_data()

                fake_input = min_data
                input_handle.reshape(min_data.shape)
                input_handle.copy_from_cpu(fake_input)
                predictor.run()
                output_names = predictor.get_output_names()
                output_handle = predictor.get_output_handle(output_names[0])
                infer_out = output_handle.copy_to_cpu()


if __name__ == "__main__":
    unittest.main()

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
from get_program import (
    get_r50_program,
    get_r50_refit_program,
)

import paddle
import paddle.inference as paddle_infer
from paddle.quantization import PTQ, QuantConfig
from paddle.quantization.observers import AbsmaxObserver
from paddle.tensorrt.export import (
    Input,
    PrecisionMode,
    TensorRTConfig,
    convert,
    convert_to_trt,
)
from paddle.tensorrt.util import (
    predict_program,
)
from paddle.vision.models import resnet18


def standardize(array):
    mean_val = np.mean(array)
    std_val = np.std(array)
    standardized_array = (array - mean_val) / std_val
    return standardized_array


class TestConverterResNet50(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, 'pir-trt')

    def test_paddle_to_tensorrt_conversion_r50(self):
        # Step1: get program and init fake inputs
        program, scope, param_dict = get_r50_program()

        # Set input
        input_config = Input(
            min_input_shape=(1, 3, 224, 224),
            optim_input_shape=(1, 3, 224, 224),
            max_input_shape=(4, 3, 224, 224),
            input_data_type='float32',
            name='input',
        )
        _, input_optim_data, _ = input_config.generate_input_data()

        # Create a TensorRTConfig with inputs as a required field.
        trt_config = TensorRTConfig(inputs=[input_config])
        trt_config.disable_passes = ['dead_code_elimination_pass']

        output_var = program.list_vars()[-1]

        # get original results(for tests only)

        output_expected = predict_program(
            program, {"input": input_optim_data}, [output_var]
        )

        program_with_trt = convert_to_trt(program, trt_config, scope)
        output_var = program_with_trt.list_vars()[-1]

        # Step6: run inference(converted_program)
        output_converted = predict_program(
            program_with_trt, {"input": input_optim_data}, [output_var]
        )

        output_expected = standardize(output_expected[0])
        output_trt = standardize(output_converted[0])

        # Check that the results are close to each other within a tolerance of 1e-3
        np.testing.assert_allclose(
            output_expected,
            output_trt,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Outputs are not within the 1e-3 tolerance",
        )

    def test_refit(self):
        # Step1: get program and init fake inputs
        paddle.enable_static()
        save_path = os.path.join(self.temp_dir.name, 'resnet50')
        program, scope, param_dict = get_r50_refit_program(save_path)

        # Set input
        input_config = Input(
            min_input_shape=(1, 3, 224, 224),
            optim_input_shape=(1, 3, 224, 224),
            max_input_shape=(4, 3, 224, 224),
            input_data_type='float32',
        )
        _, input_optim_data, _ = input_config.generate_input_data()

        # Create a TensorRTConfig with inputs as a required field.
        trt_config = TensorRTConfig(inputs=[input_config])

        output_var = program.list_vars()[-1]

        # get original results(for tests only)

        output_expected = predict_program(
            program, {"input": input_optim_data}, [output_var]
        )

        trt_save_path = os.path.join(self.temp_dir.name, 'resnet50trt')
        trt_config.save_model_dir = trt_save_path
        trt_config.refit_params_path = save_path + '.pdiparams'
        model_dir = save_path

        program_with_trt = paddle.tensorrt.convert(model_dir, trt_config)
        config = paddle_infer.Config(
            trt_config.save_model_dir + '.json',
            trt_config.save_model_dir + '.pdiparams',
        )
        config.switch_ir_debug(True)
        if paddle.is_compiled_with_cuda():
            config.enable_use_gpu(100, 0)
        else:
            config.disable_gpu()
        predictor = paddle_infer.create_predictor(config)

        paddle.disable_static()
        for i, input_instrance in enumerate(trt_config.inputs):
            min_data, _, max_data = input_instrance.generate_input_data()
            model_inputs = paddle.to_tensor(min_data)
            output_converted = predictor.run([model_inputs])

        output_expected = standardize(output_expected[0])
        output_trt = standardize(output_converted[0].numpy())

        np.testing.assert_allclose(
            output_expected,
            output_trt,
            rtol=1e-1,
            atol=1e-1,
            err_msg="Outputs are not within the 1e-1 tolerance",
        )

    def test_paddle_to_tensorrt_conversion_r50_collect_shape(self):
        # Step1: get program and init fake inputs
        program, scope, param_dict = get_r50_program()

        # Set input
        input_data = tuple(
            np.random.rand(n, 3, 224, 224).astype(np.float32) for n in (1, 2, 4)
        )
        input_optim_data = input_data[1]
        input_config = Input(warmup_data=input_data)

        # Create a TensorRTConfig with inputs as a required field.
        trt_config = TensorRTConfig(inputs=[input_config])
        trt_config.disable_passes = ['dead_code_elimination_pass']

        output_var = program.list_vars()[-1]

        # get original results(for tests only)

        output_expected = predict_program(
            program, {"input": input_optim_data}, [output_var]
        )

        program_with_trt = convert_to_trt(program, trt_config, scope)
        output_var = program_with_trt.list_vars()[-1]

        # Step6: run inference(converted_program)
        output_converted = predict_program(
            program_with_trt, {"input": input_optim_data}, [output_var]
        )

        output_expected = standardize(output_expected[0])
        output_trt = standardize(output_converted[0])

        # Check that the results are close to each other within a tolerance of 1e-3
        np.testing.assert_allclose(
            output_expected,
            output_trt,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Outputs are not within the 1e-2 tolerance",
        )

    def test_convert_quant_model(self):
        paddle.disable_static()
        image = paddle.ones([1, 3, 224, 224], dtype="float32")
        model = resnet18()
        model.eval()
        output_fp32 = model(image)

        observer = AbsmaxObserver(quant_bits=8)
        q_config = QuantConfig(activation=observer, weight=observer)
        ptq = PTQ(q_config)
        quant_model = ptq.quantize(model)
        out = quant_model(image)
        converted_model = ptq.convert(quant_model)
        save_path = os.path.join(self.temp_dir.name, 'int8_infer')
        paddle.jit.save(converted_model, save_path, input_spec=[image])

        paddle.enable_static()
        trt_save_path = os.path.join(self.temp_dir.name, 'int8_trt_infer')
        # Set input
        input_config = Input(
            min_input_shape=(1, 3, 224, 224),
            optim_input_shape=(1, 3, 224, 224),
            max_input_shape=(1, 3, 224, 224),
            input_data_type='float32',
        )
        trt_config = TensorRTConfig(inputs=[input_config])
        trt_config.disable_passes = ['dead_code_elimination_pass']
        trt_config.save_model_dir = trt_save_path
        trt_config.precision_mode = PrecisionMode.INT8
        convert(save_path, trt_config)

        config = paddle_infer.Config(
            trt_config.save_model_dir + '.json',
            trt_config.save_model_dir + '.pdiparams',
        )
        config.enable_use_gpu(100, 0)
        predictor = paddle_infer.create_predictor(config)
        output_trt_int8 = predictor.run([image])

        # Check that the results are close to each other within a tolerance of 0.9
        np.testing.assert_allclose(
            output_fp32,
            output_trt_int8[0],
            rtol=0.9,
            atol=0.9,
            err_msg="Outputs are not within the 0.9 tolerance",
        )

    def test_paddle_to_tensorrt_conversion_r50_use_cuda_graph(self):
        # Step1: get program and init fake inputs
        program, scope, param_dict = get_r50_program()

        # Set input
        input_config = Input(
            min_input_shape=(1, 3, 224, 224),
            optim_input_shape=(1, 3, 224, 224),
            max_input_shape=(4, 3, 224, 224),
            input_data_type='float32',
            name='input',
        )
        _, input_optim_data, _ = input_config.generate_input_data()

        # Create a TensorRTConfig with inputs as a required field.
        trt_config = TensorRTConfig(inputs=[input_config])
        trt_config.disable_passes = ['dead_code_elimination_pass']

        # use_cuda_graph: True
        trt_config.use_cuda_graph = True

        output_var = program.list_vars()[-1]

        # get original results(for tests only)

        output_expected = predict_program(
            program, {"input": input_optim_data}, [output_var]
        )

        program_with_trt = convert_to_trt(program, trt_config, scope)
        output_var = program_with_trt.list_vars()[-1]

        # Step6: run inference(converted_program)
        output_converted = predict_program(
            program_with_trt, {"input": input_optim_data}, [output_var]
        )

        output_expected = standardize(output_expected[0])
        output_trt = standardize(output_converted[0])

        # Check that the results are close to each other within a tolerance of 1e-3
        np.testing.assert_allclose(
            output_expected,
            output_trt,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Outputs are not within the 1e-3 tolerance",
        )


if __name__ == "__main__":
    unittest.main()

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

import unittest

import numpy as np
from get_program import (
    get_dummy_program,
)

from paddle.tensorrt.export import (
    Input,
    PrecisionMode,
    TensorRTConfig,
    convert_to_trt,
)
from paddle.tensorrt.util import (
    predict_program,
)


class TestConverterDummy(unittest.TestCase):
    def test_paddle_to_tensorrt_conversion_dummy(self):
        program, scope, param_dict = get_dummy_program()

        # Set input
        input_config = Input(
            min_input_shape=(1, 64),
            optim_input_shape=(4, 64),
            max_input_shape=(8, 64),
            input_data_type='float32',
        )
        _, input_optim_data, _ = input_config.generate_input_data()
        # Create a TensorRTConfig with inputs as a required field.
        trt_config = TensorRTConfig(inputs=[input_config])
        trt_config.precision_mode = PrecisionMode.FP16
        trt_config.ops_run_float = "pd_op.add"
        trt_config.optimization_level = 5

        output_var = program.list_vars()[-1]

        # get original results(for tests only)
        output_expected = predict_program(
            program, {"input": input_optim_data}, [output_var]
        )
        # get tensorrt_engine_op(converted_program)
        program_with_trt = convert_to_trt(program, trt_config, scope)
        output_var = program_with_trt.list_vars()[-1]

        # run inference(converted_program)
        output_converted = predict_program(
            program_with_trt, {"input": input_optim_data}, [output_var]
        )

        # Check that the results are close to each other within a tolerance of 1e-2
        np.testing.assert_allclose(
            output_expected[0],
            output_converted[0],
            rtol=1e-2,
            atol=1e-2,
            err_msg="Outputs are not within the 1e-2 tolerance",
        )


if __name__ == "__main__":
    unittest.main()

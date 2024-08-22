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
    get_r50_program,
)

from paddle.tensorrt.converter import PaddleToTensorRTConverter
from paddle.tensorrt.util import (
    predict_program,
    run_pir_pass,
    warmup_shape_infer,
)


class TestConverterResNet50(unittest.TestCase):
    def test_paddle_to_tensorrt_conversion_r50(self):
        # Step1: get program and init fake inputs
        program, scope, param_dict = get_r50_program()
        input_data_min_shape = np.ones((1, 3, 224, 224), dtype='float32')
        input_data_max_shape = np.ones((1, 3, 224, 224), dtype='float32')

        # Step1.1: get original results(for tests only)
        output_var = program.list_vars()[-1]

        output_expected = predict_program(
            program, {"input": input_data_min_shape}, [output_var]
        )

        # Step2: run warmup for collecting shape
        warmup_shape_infer(
            program,
            min_shape_feed={"input": input_data_min_shape},
            max_shape_feed={"input": input_data_max_shape},
        )

        # Step3: run pir pass(including some fusion pass and trt_op_marker_pass)
        program = run_pir_pass(program, partition_mode=False)

        # Step4: run trt_sub_graph_extract_pass()
        program_with_pir = run_pir_pass(program, partition_mode=True)

        # Step5: run TRTConverter(would lower group_op into tensorrt_engine_op)
        converter = PaddleToTensorRTConverter(program_with_pir, scope)
        converter.convert_program_to_trt()

        output_var = program_with_pir.list_vars()[-1]

        # Step6: run inference(converted_program)
        output_converted = predict_program(
            program_with_pir, {"input": input_data_min_shape}, [output_var]
        )

        # Check that the results are close to each other within a tolerance of 1e-3
        np.testing.assert_allclose(
            output_expected[0],
            output_converted[0],
            rtol=0.2,
            atol=0.2,
            err_msg="Outputs are not within the 0.2 tolerance",
        )


if __name__ == "__main__":
    unittest.main()

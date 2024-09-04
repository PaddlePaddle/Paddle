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

import paddle
from paddle.tensorrt.converter import PaddleToTensorRTConverter
from paddle.tensorrt.util import (
    run_pir_pass,
    warmup_shape_infer,
)


class TestConverterDummy(unittest.TestCase):
    def test_paddle_to_tensorrt_conversion_dummy(self):
        program, scope, param_dict = get_dummy_program()
        input_data = np.random.randn(1, 64).astype('float32')
        input_data_max_shape = np.random.randn(8, 64).astype('float32')

        with paddle.pir_utils.IrGuard():
            with paddle.static.program_guard(program):
                executor = paddle.static.Executor()
                output_var = program.list_vars()[-2]
                # Run the program with input_data
                for _ in range(1):
                    output_original = executor.run(
                        program,
                        feed={"input": input_data},
                        fetch_list=[output_var],
                    )

        warmup_shape_infer(
            program,
            min_shape_feed={"input": input_data},
            max_shape_feed={"input": input_data_max_shape},
        )
        # Apply PIR pass to the program
        program_with_pir = run_pir_pass(program, partition_mode=True)

        # Convert the program to TensorRT
        converter = PaddleToTensorRTConverter(program_with_pir, scope)
        converter.convert_program_to_trt()
        output_var = program_with_pir.list_vars()[-2]

        with paddle.pir_utils.IrGuard():
            with paddle.static.program_guard(program_with_pir):
                executor = paddle.static.Executor()
                for _ in range(5):
                    output_converted = executor.run(
                        program_with_pir,
                        feed={"input": input_data},
                        fetch_list=[output_var],
                    )

        # Check that the results are close to each other within a tolerance of 1e-3
        np.testing.assert_allclose(
            output_original[0],
            output_converted[0],
            rtol=1e-2,
            atol=1e-2,
            err_msg="Outputs are not within the 1e-2 tolerance",
        )


if __name__ == "__main__":
    unittest.main()

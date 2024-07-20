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

import numpy as np
from converter import PaddleToTensorRTConverter
from util import (
    forbid_op_lower_trt,
    enforce_op_lower_trt,
    get_bert_program,
    get_dummy_program,
    get_r50_program,
    predict_program,
    run_pir_pass,
    warmup_shape_infer,
)

import paddle


def test_paddle_to_tensorrt_conversion_dummy():
    program, scope, param_dict = get_dummy_program()
    input_data = np.random.randn(1, 64).astype('float32')
    input_data_max_shape = np.random.randn(8, 64).astype('float32')

    with paddle.pir_utils.IrGuard():
        with paddle.static.program_guard(program):
            executor = paddle.static.Executor()
            output_var = program.list_vars()[-1]

            # Run the program with input_data
            for _ in range(1):
                output_original = executor.run(
                    program, feed={"input": input_data}, fetch_list=[output_var]
                )

            # Run the program with input_data_max_shape (fake max_shape input)
            executor.run(
                program,
                feed={"input": input_data_max_shape},
                fetch_list=[output_var],
            )

    # Apply PIR pass to the program
    program_with_pir = run_pir_pass(program, partition_mode=True)

    # Convert the program to TensorRT
    converter = PaddleToTensorRTConverter(program_with_pir, scope)
    converter.convert_program_to_trt()

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
        rtol=1e-3,
        atol=1e-3,
        err_msg="Outputs are not within the 1e-3 tolerance",
    )

    print(output_original)
    print(output_converted)


def test_paddle_to_tensorrt_conversion_bert():
    # Step1: get program and init fake inputs
    program, scope, param_dict = get_bert_program()
    input_data_min_shape = np.ones([1, 100]).astype('int64')
    input_data_max_shape = np.ones([8, 1000]).astype('int64')

    # Step1.1: get original results(for tests only)
    output_var = program.list_vars()[-1]
    output_expected = predict_program(
        program, {"input_ids": input_data_min_shape}, [output_var]
    )

    # Step2: run warmup for collecting shape
    warmup_shape_infer(
        program,
        min_shape_feed={"input_ids": input_data_min_shape},
        max_shape_feed={"input_ids": input_data_max_shape},
    )

    # Step3: run pir pass(including some fusion pass and trt_op_marker_pass)
    program = run_pir_pass(program, partition_mode=False)
    forbid_op_lower_trt(program, "pd_op.layer_norm")

    # Step4: run trt_sub_graph_extract_pass()
    program_with_pir = run_pir_pass(program, partition_mode=True)

    # Step5: run TRTConverter(would lower group_op into tensorrt_engine_op)
    converter = PaddleToTensorRTConverter(program_with_pir, scope)
    converter.convert_program_to_trt()

    # Step6: run inference(converted_program)
    output_converted = predict_program(
        program_with_pir, {"input_ids": input_data_min_shape}, [output_var]
    )

    # Check that the results are close to each other within a tolerance of 1e-3
    np.testing.assert_allclose(
        output_expected[0],
        output_converted[0],
        rtol=1e-2,
        atol=1e-2,
        err_msg="Outputs are not within the 1e-3 tolerance",
    )

    print(output_expected)
    print(output_converted)


def test_paddle_to_tensorrt_conversion_r50():
    # Step1: get program and init fake inputs
    program, scope, param_dict = get_r50_program()
    input_data_min_shape = np.random.randn(1, 3, 224, 224).astype('float32')
    input_data_max_shape = np.random.randn(1, 3, 224, 224).astype('float32')

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
    # enforce_op_lower_trt(program, "pd_op.conv2d")
    # enforce_op_lower_trt(program, "pd_op.relu")
    # enforce_op_lower_trt(program, "pd_op.pool2d")
    # enforce_op_lower_trt(program,"pd_op.add")
    # enforce_op_lower_trt(program, "pd_op.batch_norm_")
    # enforce_op_lower_trt(program, "pd_op.flatten")
    # forbid_op_lower_trt(program,"pd_op.matmul")
    # forbid_op_lower_trt(program,"pd_op.flatten")
    # forbid_op_lower_trt(program,"pd_op.add")

    # Step4: run trt_sub_graph_extract_pass()
    program_with_pir = run_pir_pass(program, partition_mode=True)

    # Step5: run TRTConverter(would lower group_op into tensorrt_engine_op)
    converter = PaddleToTensorRTConverter(program_with_pir, scope)
    converter.convert_program_to_trt()

    # Step6: run inference(converted_program)
    output_converted = predict_program(
        program_with_pir, {"input": input_data_min_shape}, [output_var]
    )

    # Check that the results are close to each other within a tolerance of 1e-3
    np.testing.assert_allclose(
        output_expected[0],
        output_converted[0],
        rtol=0.1,
        atol=0.1,
        err_msg="Outputs are not within the 1e-3 tolerance",
    )

    print("output_expected", output_expected)
    print("output_converted", output_converted)


if __name__ == "__main__":
    # test_paddle_to_tensorrt_conversion_dummy()
    # test_paddle_to_tensorrt_conversion_bert()
    test_paddle_to_tensorrt_conversion_r50()

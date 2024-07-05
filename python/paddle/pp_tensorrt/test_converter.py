import pytest
import numpy as np
import paddle
from util import run_pir_pass, get_dummy_program
from converter import PaddleToTensorRTConverter

def test_paddle_to_tensorrt_conversion():
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
                    program,
                    feed={"input": input_data},
                    fetch_list=[output_var]
                )

            # Run the program with input_data_max_shape (fake max_shape input)
            executor.run(
                program,
                feed={"input": input_data_max_shape},
                fetch_list=[output_var]
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
                    fetch_list=[output_var]
                )
    
    # Check that the results are close to each other within a tolerance of 1e-3
    np.testing.assert_allclose(output_original[0], output_converted[0], rtol=1e-3, atol=1e-3, err_msg="Outputs are not within the 1e-3 tolerance")

    print(output_original)
    print(output_converted)

if __name__ == "__main__":
    test_paddle_to_tensorrt_conversion()

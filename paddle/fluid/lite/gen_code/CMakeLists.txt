lite_cc_library(gen_code_lite SRCS gen_code.cc
        DEPS program_lite op_lite scope_lite
        cpp_op_desc_lite
        HVY_DEPS operator)
lite_cc_library(paddle_infer_gencode SRCS paddle_infer.cc DEPS program_lite utils_lite)

lite_cc_test(test_gen_code_lite SRCS gen_code_test.cc 
        DEPS gen_code_lite ${tensor_lite} ${host_kernels} ${ops_lite} 
        compatible_pb_lite
        model_parser_lite
        X86_DEPS ${x86_kernels}
        ARM_DEPS ${arm_kernels}
        CL_DEPS ${opencl_kernels}
        EXCLUDE_COMPILE_DEPS "ON"
        ARGS --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)

lite_cc_library(__generated_code__
    SRCS ${CMAKE_BINARY_DIR}/paddle/fluid/lite/gen_code/__generated_code__.cc
    DEPS scope_lite op_lite kernel_lite paddle_infer_gencode
    EXCLUDE_COMPILE_DEPS "ON"
)
add_dependencies(__generated_code__ test_gen_code_lite)
add_dependencies(__generated_code__ extern_lite_download_lite_naive_model_tar_gz)

lite_cc_test(test_generated_code SRCS generated_code_test.cc DEPS __generated_code__
    ${ops_lite} ${host_kernels}
    X86_DEPS ${x86_kernels}
    ARM_DEPS ${arm_kernels}
    CL_DEPS ${opencl_kernels}
    EXCLUDE_COMPILE_DEPS "ON"
)

lite_cc_binary(paddle_code_generator SRCS paddle_code_generator.cc DEPS model_parser_lite gen_code_lite)

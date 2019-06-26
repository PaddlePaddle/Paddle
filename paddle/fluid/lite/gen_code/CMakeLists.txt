lite_cc_library(gen_code_lite SRCS gen_code.cc
        DEPS program_lite op_lite scope_lite
        cpp_op_desc_lite
        HVY_DEPS operator)
lite_cc_library(paddle_infer_gencode SRCS paddle_infer.cc DEPS program_lite utils_lite)

if (NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    lite_cc_test(test_gen_code_lite SRCS gen_code_test.cc DEPS gen_code_lite ${tensor_lite}
            mul_op_lite
            compatible_pb_lite
            model_parser_lite
            X86_DEPS mul_compute_x86
            ARM_DEPS mul_compute_arm
            ARGS --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)

    lite_cc_library(__generated_code__
        SRCS ${CMAKE_BINARY_DIR}/paddle/fluid/lite/gen_code/__generated_code__.cc
        DEPS scope_lite op_lite kernel_lite paddle_infer_gencode
    )

    lite_cc_test(test_generated_code SRCS generated_code_test.cc DEPS __generated_code__
     ${ops_lite} ${host_kernels}
     X86_DEPS ${x86_kernels}
    )

    add_dependencies(__generated_code__ test_gen_code_lite)
    add_dependencies(__generated_code__ extern_lite_download_lite_naive_model_tar_gz)
endif()

lite_cc_binary(paddle_code_generator SRCS paddle_code_generator.cc DEPS model_parser_lite gen_code_lite)

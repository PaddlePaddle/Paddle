core_gather_headers()

gather_srcs(infrt_src SRCS
    kernel_frame.cc
    kernel_registry.cc
    value.cc
    kernel_utils.cc
    symbol_table.cc
    op_executable.cc
    core_runtime.cc
    mlir_to_runtime_translate.cc
    function.cc
    mlir_function_executable.cc
    mlir_program_executor.cc
    )

cc_test_tiny(test_infrt_host_context_value SRCS value_test.cc DEPS infrt ${MLIR_IR_LIBS})
cc_test_tiny(test_infrt_kernel_utils SRCS kernel_utils_test.cc DEPS infrt ${MLIR_IR_LIBS})
cc_test_tiny(test_infrt_kernel_registry SRCS kernel_registry_test.cc DEPS infrt ${MLIR_IR_LIBS})
cc_test_tiny(test_infrt_op_executable SRCS op_executable_test.cc DEPS infrt ${MLIR_IR_LIBS})
cc_test_tiny(test_infrt_core_runtime SRCS core_runtime_test.cc DEPS infrt ${MLIR_IR_LIBS})
cc_test_tiny(test_infrt_mlir_to_runtime_translate SRCS mlir_to_runtime_translate_test.cc DEPS infrt ${MLIR_IR_LIBS})

infrt_exec_check(test_infrt_mlir_exec_on_basic mlir_tests/basic.mlir)
infrt_exec_check(test_infrt_mlir_exec_on_shape mlir_tests/shape.mlir)
infrt_exec_check(test_infrt_mlir_exec_on_dense_tensor mlir_tests/dense_tensor.mlir)

add_executable(infrt-exec mlir_exec.cc)
target_link_libraries(infrt-exec infrt ${MLIR_IR_LIBS})

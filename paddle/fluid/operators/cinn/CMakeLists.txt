include(operators)
register_operators(EXCLUDES cinn_launch_op)

cc_library(cinn_launch_context SRCS cinn_launch_context.cc DEPS cinn)
op_library(cinn_launch_op SRCS cinn_launch_op.cc cinn_launch_op.cu.cc DEPS cinn cinn_compiler cinn_launch_context)

if (WITH_TESTING)
  cc_test(cinn_launch_context_test SRCS cinn_launch_context_test.cc DEPS scope lod_tensor cinn_launch_context)
  set_tests_properties(cinn_launch_context_test PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test(cinn_launch_op_test SRCS cinn_launch_op_test.cc DEPS cinn_compiler cinn_launch_op elementwise_add_op)
  set_tests_properties(cinn_launch_op_test PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;runtime_include_dir=${PADDLE_BINARY_DIR}/third_party/CINN/src/external_cinn/cinn/runtime/cuda")
endif()

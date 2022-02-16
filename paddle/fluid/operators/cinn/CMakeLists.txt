include(operators)

cc_library(cinn_op_helper SRCS cinn_op_helper.cc DEPS operator device_context)
cc_library(cinn_launch_context SRCS cinn_launch_context.cc DEPS ddim lod_tensor scope cinn)

SET(CINN_OP_DEPS string_helper cinn cinn_compiler cinn_op_helper cinn_launch_context)
register_operators(DEPS ${CINN_OP_DEPS})

if (WITH_TESTING)
  cc_test(cinn_launch_context_test SRCS cinn_launch_context_test.cc DEPS ddim lod_tensor scope cinn_launch_context)
  set_tests_properties(cinn_launch_context_test PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test(cinn_launch_op_test SRCS cinn_launch_op_test.cc DEPS cinn_compiler cinn_launch_op elementwise_add_op)
  set_tests_properties(cinn_launch_op_test PROPERTIES LABELS "RUN_TYPE=CINN" ENVIRONMENT "OMP_NUM_THREADS=1;runtime_include_dir=${PADDLE_BINARY_DIR}/third_party/CINN/src/external_cinn/cinn/runtime/cuda")
endif()

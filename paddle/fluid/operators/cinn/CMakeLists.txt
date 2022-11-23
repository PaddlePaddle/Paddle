include(operators)

cc_library(
  cinn_op_helper
  SRCS cinn_op_helper.cc
  DEPS operator device_context)
cc_library(
  cinn_launch_context
  SRCS cinn_launch_context.cc
  DEPS ddim
       lod_tensor
       scope
       proto_desc
       graph
       build_strategy
       device_context
       parallel_executor
       transform_type
       cinn)

set(CINN_OP_DEPS
    parallel_executor
    string_helper
    variable_helper
    cinn
    cinn_compiler
    cinn_op_helper
    cinn_launch_context
    transform_type)
register_operators(DEPS ${CINN_OP_DEPS})

if(WITH_TESTING)
<<<<<<< HEAD
  cc_test(
    cinn_launch_context_test
    SRCS cinn_launch_context_test.cc
    DEPS ddim
         lod_tensor
         scope
         proto_desc
         graph
         cinn_launch_context
         cinn_instruction_run_op
         cinn)
=======
  cc_test_old(
    cinn_launch_context_test
    SRCS
    cinn_launch_context_test.cc
    DEPS
    ddim
    lod_tensor
    scope
    proto_desc
    graph
    cinn_launch_context
    cinn_instruction_run_op
    cinn)
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  set_tests_properties(cinn_launch_context_test PROPERTIES LABELS
                                                           "RUN_TYPE=CINN")

  set(CINN_RUN_ENVIRONMENT
      "OMP_NUM_THREADS=1;runtime_include_dir=${PADDLE_BINARY_DIR}/third_party/CINN/src/external_cinn/cinn/runtime/cuda"
  )
<<<<<<< HEAD
  cc_test(
    cinn_launch_op_test
    SRCS cinn_launch_op_test.cc
    DEPS cinn_compiler cinn_launch_op cinn_instruction_run_op
         elementwise_add_op gflags)
=======
  cc_test_old(
    cinn_launch_op_test
    SRCS
    cinn_launch_op_test.cc
    DEPS
    cinn_compiler
    cinn_launch_op
    cinn_instruction_run_op
    elementwise_add_op
    gflags)
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  set_tests_properties(
    cinn_launch_op_test PROPERTIES LABELS "RUN_TYPE=CINN" ENVIRONMENT
                                   "${CINN_RUN_ENVIRONMENT}")

<<<<<<< HEAD
  cc_test(
    cinn_instruction_run_op_test
    SRCS cinn_instruction_run_op_test.cc
    DEPS cinn_compiler cinn_launch_op cinn_instruction_run_op
         elementwise_add_op)
=======
  cc_test_old(
    cinn_instruction_run_op_test
    SRCS
    cinn_instruction_run_op_test.cc
    DEPS
    cinn_compiler
    cinn_launch_op
    cinn_instruction_run_op
    elementwise_add_op)
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  set_tests_properties(
    cinn_instruction_run_op_test PROPERTIES LABELS "RUN_TYPE=CINN" ENVIRONMENT
                                            "${CINN_RUN_ENVIRONMENT}")
endif()

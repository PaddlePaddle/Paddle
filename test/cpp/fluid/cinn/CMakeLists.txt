if(WITH_TESTING)
  cc_test_old(
    cinn_launch_context_test
    SRCS
    cinn_launch_context_test.cc
    DEPS
    fleet_executor
    phi
    lod_tensor
    scope
    proto_desc
    graph
    cinn_launch_context
    cinn_instruction_run_op
    cinnapi
    paddle_flags)
  target_link_libraries(cinn_launch_context_test ${PYTHON_LIBRARIES})
  set_tests_properties(cinn_launch_context_test PROPERTIES LABELS
                                                           "RUN_TYPE=CINN")

  set(CINN_RUN_ENVIRONMENT
      "OMP_NUM_THREADS=1;runtime_include_dir=${CINN_INCLUDE_DIR}/paddle/cinn/runtime/cuda/"
  )
  # cc_test_old(
  #   cinn_launch_op_test
  #   SRCS
  #   cinn_launch_op_test.cc
  #   DEPS
  #   cinn_compiler
  #   cinn_launch_op
  #   cinn_instruction_run_op
  #   elementwise_add_op
  #   gflags)
  # set_tests_properties(
  #   cinn_launch_op_test PROPERTIES LABELS "RUN_TYPE=CINN" ENVIRONMENT
  #                                  "${CINN_RUN_ENVIRONMENT}")

  cc_test_old(
    cinn_instruction_run_op_test
    SRCS
    cinn_instruction_run_op_test.cc
    DEPS
    cinn_compiler
    cinn_launch_op
    cinn_instruction_run_op
    elementwise_add_op
    paddle_flags)
  target_link_libraries(cinn_instruction_run_op_test ${PYTHON_LIBRARIES})

  get_property(
    env
    TEST cinn_instruction_run_op_test
    PROPERTY ENVIRONMENT)
  set_property(TEST cinn_instruction_run_op_test
               PROPERTY ENVIRONMENT "${CINN_RUN_ENVIRONMENT}" ${env})
  set_tests_properties(cinn_instruction_run_op_test PROPERTIES LABELS
                                                               "RUN_TYPE=CINN")
endif()

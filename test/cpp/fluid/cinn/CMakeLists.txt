if(WITH_TESTING)
  paddle_test(cinn_launch_context_test SRCS cinn_launch_context_test.cc)
  target_link_libraries(cinn_launch_context_test ${PYTHON_LIBRARIES})
  set_tests_properties(cinn_launch_context_test PROPERTIES LABELS
                                                           "RUN_TYPE=CINN")

  set(CINN_RUN_ENVIRONMENT
      "OMP_NUM_THREADS=1;runtime_include_dir=${CINN_INCLUDE_DIR}/paddle/cinn/runtime/cuda/"
  )

endif()

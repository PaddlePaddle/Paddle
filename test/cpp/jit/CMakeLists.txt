if(WITH_TESTING AND NOT WIN32)
  add_custom_target(
    jit_download_program
    COMMAND
      wget -nc -q --no-check-certificate
      https://paddle-ci.gz.bcebos.com/dy2st/multi_program_load_with_property.tar.gz
    COMMAND tar zxf multi_program_load_with_property.tar.gz
    WORKING_DIRECTORY "${CC_TESTS_DIR}")
  set(JIT_DEPS
      phi
      common
      elementwise_add_op
      activation_op
      reduce_mean_op
      feed_op
      fetch_op
      generated_op
      transfer_layout_op
      fluid_jit
      generated_static_op)
  cc_test(
    layer_test
    SRCS layer_test.cc
    DEPS ${JIT_DEPS})
endif()

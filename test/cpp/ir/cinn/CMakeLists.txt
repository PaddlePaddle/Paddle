if(WITH_TESTING AND WITH_CINN)
  cc_test_old(
    test_new_ir_compiler
    SRCS
    new_ir_compiler_test.cc
    DEPS
    new_ir_compiler
    ir
    phi
    gtest
    glog)
  set_tests_properties(test_new_ir_compiler PROPERTIES LABELS "RUN_TYPE=CINN")
endif()

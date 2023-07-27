if(WITH_TESTING AND WITH_CINN)
  cc_test_old(
    test_new_ir_compiler
    SRCS
    new_ir_compiler_test.cc
    DEPS
    cinncore
    pd_dialect
    ir
    phi
    gtest
    glog)
  set_tests_properties(test_new_ir_compiler PROPERTIES LABELS "RUN_TYPE=CINN")
endif()

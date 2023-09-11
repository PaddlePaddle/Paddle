cc_test_old(
  symbolic_op_test
  SRCS
  symbolic_op_test.cc
  DEPS
  pd_op_dialect
  pir
  gtest)

set(TEST_DEPS gtest pd_op_dialect pir)

if(WITH_DISTRIBUTE)
  set(TEST_DEPS ${TEST_DEPS} fleet_executor)
endif()

if(WIN32 AND WITH_TESTING)
  cc_test_old(
    constraint_pass_test
    SRCS
    constraint_pass_test.cc
    DEPS
    gtest
    ${TEST_DEPS}
    ${BRPC_DEPS})
else()
  cc_test_old(
    constraint_pass_test
    SRCS
    constraint_pass_test.cc
    DEPS
    gtest
    ${TEST_DEPS}
    ${paddle_lib}
    python)
endif()

set_tests_properties(
  constraint_pass_test PROPERTIES ENVIRONMENT
                                  "FLAGS_enable_new_ir_in_executor=true")

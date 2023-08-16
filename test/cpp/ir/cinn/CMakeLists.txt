if(WITH_TESTING AND WITH_CINN)
  cc_test_old(
    test_new_ir_compiler
    SRCS
    new_ir_compiler_test.cc
    DEPS
    new_ir_compiler
    convert_to_dialect
    ir
    phi
    gtest
    glog)
  set_tests_properties(test_new_ir_compiler PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test_old(
    test_jit_instruction
    SRCS
    jit_instruction_test.cc
    DEPS
    interpreter
    new_ir_compiler
    convert_to_dialect)
  set_tests_properties(test_jit_instruction PROPERTIES LABELS "RUN_TYPE=CINN")
endif()

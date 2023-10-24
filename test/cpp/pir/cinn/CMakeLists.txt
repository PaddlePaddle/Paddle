if(WITH_TESTING AND WITH_CINN)
  paddle_test(test_pir_compiler SRCS pir_compiler_test.cc DEPS pir_compiler
              cinn_runtime_dialect)
  set_tests_properties(test_pir_compiler PROPERTIES LABELS "RUN_TYPE=CINN")

  paddle_test(test_jit_instruction SRCS jit_instruction_test.cc DEPS
              cinn_runtime_dialect pir_compiler)
  set_tests_properties(test_jit_instruction PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test_old(
    ir_op_fusion_test
    SRCS
    ir_op_fusion_test.cc
    DEPS
    op_with_group_merge_pass
    pd_op_dialect
    cinn_op_dialect
    pir
    gtest
    glog)
  set_tests_properties(ir_op_fusion_test PROPERTIES LABELS "RUN_TYPE=CINN")

  paddle_test(test_group_op SRCS group_op_test.cc DEPS op_with_group_merge_pass
              cinn_op_dialect)
  set_tests_properties(test_group_op PROPERTIES LABELS "RUN_TYPE=CINN")

  paddle_test(test_pir_build_cinn_pass SRCS build_cinn_pass_test.cc DEPS
              pd_build_cinn_pass pir_pass)
  set_tests_properties(test_pir_build_cinn_pass PROPERTIES LABELS
                                                           "RUN_TYPE=CINN")
endif()

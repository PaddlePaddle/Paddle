if(WITH_TESTING AND WITH_CINN)
  cc_test_old(
    test_graph_compiler_new_ir
    SRCS
    graph_compiler_new_ir_test.cc
    DEPS
    cinncore
    pd_dialect
    ir
    phi
    gtest
    glog)
  set_tests_properties(test_graph_compiler_new_ir PROPERTIES LABELS
                                                             "RUN_TYPE=CINN")
endif()

cc_test_old(
  ir_op_fusion_test
  SRCS
  ir_op_fusion_test.cc
  DEPS
  general_fusion_merge_pass
  fusion_merge_pass
  pd_dialect
  ir
  gtest
  glog)

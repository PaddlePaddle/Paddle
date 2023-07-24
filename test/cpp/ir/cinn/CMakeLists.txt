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

if(WITH_TESTING AND WITH_CINN)
  paddle_test(
    test_sub_graph_checker
    SRCS
    sub_graph_checker_test.cc
    DEPS
    op_with_group_merge_pass
    pir_transforms
    cinn_op_dialect
    pd_to_cinn_pass
    add_broadcast_to_elementwise_pass
    sub_graph_checker)

  set_tests_properties(test_sub_graph_checker PROPERTIES LABELS "RUN_TYPE=CINN")

endif()

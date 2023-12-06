if(WITH_TESTING AND WITH_CINN)
  paddle_test(
    map_expr_test
    SRCS
    map_expr_test.cc
    DEPS
    op_with_group_merge_pass
    op_dialect_vjp
    cinn_op_dialect
    pir)
  set_tests_properties(map_expr_test PROPERTIES LABELS "RUN_TYPE=CINN")

endif()

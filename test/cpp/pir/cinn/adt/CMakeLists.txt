if(WITH_TESTING AND WITH_CINN)
  paddle_test(map_expr_test SRCS map_expr_test.cc)
  set_tests_properties(map_expr_test PROPERTIES LABELS "RUN_TYPE=CINN")

endif()

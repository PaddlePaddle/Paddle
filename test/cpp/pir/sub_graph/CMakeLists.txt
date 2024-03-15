if(WITH_TESTING AND WITH_CINN)
  paddle_test(test_sub_graph_checker SRCS sub_graph_checker_test.cc DEPS
              sub_graph_checker)

  set_tests_properties(test_sub_graph_checker PROPERTIES LABELS "RUN_TYPE=CINN")

endif()

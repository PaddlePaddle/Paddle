if(WITH_CINN)
  cc_library(
    sub_graph_checker
    SRCS sub_graph_checker.cc
    DEPS standalone_executor op_dialect pir cinn_op_dialect cinnapi)

endif()

if(WITH_GPU AND WITH_DGC)
  list(APPEND ALL_REDUCE_OP_HANDLES detail_op_handle)
endif()

cc_library(
  multi_devices_graph_pass
  SRCS multi_devices_graph_pass.cc
  DEPS detail_op_handle ${ALL_REDUCE_OP_HANDLES})

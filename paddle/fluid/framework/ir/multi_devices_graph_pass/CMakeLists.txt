cc_library(
  modify_op_lock_and_record_event_pass
  SRCS modify_op_lock_and_record_event_pass.cc
  DEPS detail_op_handle op_graph_view)

if(WITH_GPU AND WITH_DGC)
  list(APPEND ALL_REDUCE_OP_HANDLES detail_op_handle)
endif()

cc_library(
  multi_devices_graph_pass
  SRCS multi_devices_graph_pass.cc
  DEPS detail_op_handle ${ALL_REDUCE_OP_HANDLES})
cc_library(
  sequential_execution_pass
  SRCS sequential_execution_pass.cc
  DEPS graph graph_helper pass)

cc_library(
  add_reader_dependency_pass
  SRCS add_reader_dependency_pass.cc
  DEPS graph graph_helper pass)
cc_library(
  fix_op_run_order_pass
  SRCS fix_op_run_order_pass.cc
  DEPS graph graph_helper pass)

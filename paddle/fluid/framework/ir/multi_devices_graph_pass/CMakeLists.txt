cc_library(
  modify_op_lock_and_record_event_pass
  SRCS modify_op_lock_and_record_event_pass.cc
  DEPS detail_op_handle op_graph_view)

cc_library(
  multi_devices_graph_print_pass
  SRCS multi_devices_graph_print_pass.cc
  DEPS detail_op_handle)
cc_library(
  multi_devices_graph_check_pass
  SRCS multi_devices_graph_check_pass.cc
  DEPS detail_op_handle)

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
  set_reader_device_info_utils
  SRCS set_reader_device_info_utils.cc
  DEPS graph graph_helper pass multi_devices_graph_pass)

cc_library(
  fuse_all_reduce_op_pass
  SRCS fuse_all_reduce_op_pass.cc
  DEPS graph graph_helper)
cc_library(
  all_reduce_deps_pass
  SRCS all_reduce_deps_pass.cc
  DEPS graph graph_helper pass)
cc_library(
  backward_optimizer_op_deps_pass
  SRCS backward_optimizer_op_deps_pass.cc
  DEPS graph graph_helper pass)
cc_library(
  add_reader_dependency_pass
  SRCS add_reader_dependency_pass.cc
  DEPS graph graph_helper pass)
cc_library(
  fix_op_run_order_pass
  SRCS fix_op_run_order_pass.cc
  DEPS graph graph_helper pass)

if(NOT WITH_PSCORE)
  return()
endif()
paddle_test(
  table_test
  SRCS
  table_test.cc
  DEPS
  common_table
  table
  ps_framework_proto)

paddle_test(
  dense_table_test
  SRCS
  dense_table_test.cc
  DEPS
  common_table
  table
  ps_framework_proto)

paddle_test(
  barrier_table_test
  SRCS
  barrier_table_test.cc
  DEPS
  common_table
  table
  ps_framework_proto)

paddle_test(
  brpc_service_dense_sgd_test
  SRCS
  brpc_service_dense_sgd_test.cc
  DEPS
  scope
  ps_service
  table
  ps_framework_proto)

paddle_test(
  brpc_service_sparse_sgd_test
  SRCS
  brpc_service_sparse_sgd_test.cc
  DEPS
  scope
  ps_service
  table
  ps_framework_proto)

paddle_test(
  brpc_utils_test
  SRCS
  brpc_utils_test.cc
  DEPS
  brpc_utils
  scope
  sendrecv_rpc
  ps_service)

paddle_test(
  graph_node_test
  SRCS
  graph_node_test.cc
  DEPS
  scope
  ps_service
  table
  ps_framework_proto)

paddle_test(
  graph_node_split_test
  SRCS
  graph_node_split_test.cc
  DEPS
  scope
  ps_service
  table
  ps_framework_proto)

paddle_test(graph_table_sample_test SRCS graph_table_sample_test.cc DEPS table
            ps_framework_proto)

paddle_test(
  feature_value_test
  SRCS
  feature_value_test.cc
  DEPS
  table
  common_table
  sendrecv_rpc)

paddle_test(sparse_sgd_rule_test SRCS sparse_sgd_rule_test.cc DEPS table)

paddle_test(ctr_accessor_test SRCS ctr_accessor_test.cc DEPS table)

paddle_test(ctr_dymf_accessor_test SRCS ctr_dymf_accessor_test.cc DEPS table)

paddle_test(memory_sparse_table_test SRCS memory_sparse_table_test.cc DEPS
            table)

paddle_test(memory_sparse_geo_table_test SRCS memory_geo_table_test.cc DEPS
            table)

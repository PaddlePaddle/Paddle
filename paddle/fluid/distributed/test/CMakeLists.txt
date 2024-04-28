set_source_files_properties(
  table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  table_test
  SRCS table_test.cc
  DEPS common_table table ps_framework_proto ${COMMON_DEPS} ${RPC_DEPS})

set_source_files_properties(
  dense_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  dense_table_test
  SRCS dense_table_test.cc
  DEPS common_table table ps_framework_proto ${COMMON_DEPS} ${RPC_DEPS})

set_source_files_properties(
  barrier_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  barrier_table_test
  SRCS barrier_table_test.cc
  DEPS common_table table ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(
  brpc_service_dense_sgd_test.cc PROPERTIES COMPILE_FLAGS
                                            ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  brpc_service_dense_sgd_test
  SRCS brpc_service_dense_sgd_test.cc
  DEPS scope ps_service table ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(
  brpc_service_sparse_sgd_test.cc PROPERTIES COMPILE_FLAGS
                                             ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  brpc_service_sparse_sgd_test
  SRCS brpc_service_sparse_sgd_test.cc
  DEPS scope ps_service table ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(
  brpc_utils_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  brpc_utils_test
  SRCS brpc_utils_test.cc
  DEPS brpc_utils
       scope
       phi
       common
       sendrecv_rpc
       ps_service
       ${COMMON_DEPS}
       ${RPC_DEPS})

set_source_files_properties(
  graph_node_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  graph_node_test
  SRCS graph_node_test.cc
  DEPS scope ps_service table ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(
  graph_node_split_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  graph_node_split_test
  SRCS graph_node_split_test.cc
  DEPS scope ps_service table ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(
  graph_table_sample_test.cc PROPERTIES COMPILE_FLAGS
                                        ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  graph_table_sample_test
  SRCS graph_table_sample_test.cc
  DEPS table ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(
  feature_value_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

cc_test(
  feature_value_test
  SRCS feature_value_test.cc
  DEPS table common_table sendrecv_rpc ${COMMON_DEPS})

set_source_files_properties(
  sparse_sgd_rule_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  sparse_sgd_rule_test
  SRCS sparse_sgd_rule_test.cc
  DEPS ${COMMON_DEPS} table)

set_source_files_properties(
  ctr_accessor_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  ctr_accessor_test
  SRCS ctr_accessor_test.cc
  DEPS ${COMMON_DEPS} table)
set_source_files_properties(
  ctr_dymf_accessor_test.cc PROPERTIES COMPILE_FLAGS
                                       ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  ctr_dymf_accessor_test
  SRCS ctr_dymf_accessor_test.cc
  DEPS ${COMMON_DEPS} table)

set_source_files_properties(
  memory_sparse_table_test.cc PROPERTIES COMPILE_FLAGS
                                         ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  memory_sparse_table_test
  SRCS memory_sparse_table_test.cc
  DEPS ${COMMON_DEPS} table)

set_source_files_properties(
  memory_geo_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(
  memory_sparse_geo_table_test
  SRCS memory_geo_table_test.cc
  DEPS ${COMMON_DEPS} table)

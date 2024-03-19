set(DISTRIBUTE_COMPILE_FLAGS
    "-Wno-error=unused-value -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=parentheses -Wno-error=unused-result"
)

if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()

get_property(RPC_DEPS GLOBAL PROPERTY RPC_DEPS)

set_source_files_properties(
  table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(table_test SRCS table_test.cc DEPS ${RPC_DEPS})

set_source_files_properties(
  dense_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(dense_table_test SRCS dense_table_test.cc DEPS ${RPC_DEPS})

set_source_files_properties(
  barrier_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(barrier_table_test SRCS barrier_table_test.cc)

set_source_files_properties(
  brpc_service_dense_sgd_test.cc PROPERTIES COMPILE_FLAGS
                                            ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(brpc_service_dense_sgd_test SRCS brpc_service_dense_sgd_test.cc
            DEPS scope)

set_source_files_properties(
  brpc_service_sparse_sgd_test.cc PROPERTIES COMPILE_FLAGS
                                             ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(brpc_service_sparse_sgd_test SRCS brpc_service_sparse_sgd_test.cc
            DEPS scope)

set_source_files_properties(
  brpc_utils_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(
  brpc_utils_test
  SRCS
  brpc_utils_test.cc
  DEPS
  scope
  phi
  common
  ${RPC_DEPS})

set_source_files_properties(
  graph_node_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(graph_node_test SRCS graph_node_test.cc DEPS scope)

set_source_files_properties(
  graph_node_split_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(graph_node_split_test SRCS graph_node_split_test.cc DEPS scope)

set_source_files_properties(
  graph_table_sample_test.cc PROPERTIES COMPILE_FLAGS
                                        ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(graph_table_sample_test SRCS graph_table_sample_test.cc)

set_source_files_properties(
  feature_value_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

paddle_test(feature_value_test SRCS feature_value_test.cc)

set_source_files_properties(
  sparse_sgd_rule_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(sparse_sgd_rule_test SRCS sparse_sgd_rule_test.cc)

set_source_files_properties(
  ctr_accessor_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(ctr_accessor_test SRCS ctr_accessor_test.cc)
set_source_files_properties(
  ctr_dymf_accessor_test.cc PROPERTIES COMPILE_FLAGS
                                       ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(ctr_dymf_accessor_test SRCS ctr_dymf_accessor_test.cc)

set_source_files_properties(
  memory_sparse_table_test.cc PROPERTIES COMPILE_FLAGS
                                         ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(memory_sparse_table_test SRCS memory_sparse_table_test.cc DEPS)

set_source_files_properties(
  memory_geo_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(memory_sparse_geo_table_test SRCS memory_geo_table_test.cc)

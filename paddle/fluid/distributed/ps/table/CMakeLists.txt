set_property(GLOBAL PROPERTY TABLE_DEPS string_helper)
set(graphDir graph)
get_property(TABLE_DEPS GLOBAL PROPERTY TABLE_DEPS)
set_source_files_properties(${graphDir}/graph_edge.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_library(graph_edge SRCS ${graphDir}/graph_edge.cc)
set_source_files_properties(${graphDir}/graph_weighted_sampler.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_library(WeightedSampler SRCS ${graphDir}/graph_weighted_sampler.cc DEPS graph_edge)
set_source_files_properties(${graphDir}/graph_node.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_library(graph_node SRCS ${graphDir}/graph_node.cc DEPS WeightedSampler)
set_source_files_properties(common_dense_table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(common_sparse_table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ssd_sparse_table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(sparse_geo_table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(barrier_table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(common_graph_table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

get_property(RPC_DEPS GLOBAL PROPERTY RPC_DEPS)

set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/")
include_directories(${PADDLE_LIB_THIRD_PARTY_PATH}libmct/src/extern_libmct/libmct/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(EXTERN_DEP "")
if(WITH_HETERPS)
    set(TABLE_SRC common_sparse_table.cc ssd_sparse_table.cc common_dense_table.cc sparse_geo_table.cc barrier_table.cc common_graph_table.cc)
    set(EXTERN_DEP rocksdb)
else()
    set(TABLE_SRC common_sparse_table.cc common_dense_table.cc sparse_geo_table.cc barrier_table.cc common_graph_table.cc)
endif()

cc_library(common_table SRCS ${TABLE_SRC} DEPS ${TABLE_DEPS}
${RPC_DEPS} graph_edge graph_node device_context string_helper
simple_threadpool xxhash generator ${EXTERN_DEP})

set_source_files_properties(tensor_accessor.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(tensor_table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

cc_library(tensor_accessor SRCS tensor_accessor.cc DEPS ${TABLE_DEPS} eigen3 ps_framework_proto device_context)
cc_library(tensor_table SRCS tensor_table.cc DEPS eigen3 ps_framework_proto executor scope device_context tensor ${TABLE_DEPS})
set_source_files_properties(table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

set_source_files_properties(sparse_sgd_rule.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ctr_accessor.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(memory_sparse_table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_library(sparse_sgd_rule SRCS sparse_sgd_rule.cc DEPS ${TABLE_DEPS} ps_framework_proto)
cc_library(ctr_accessor SRCS ctr_accessor.cc DEPS ${TABLE_DEPS} ps_framework_proto sparse_sgd_rule)
cc_library(memory_sparse_table SRCS memory_sparse_table.cc DEPS ps_framework_proto ${TABLE_DEPS} fs afs_wrapper ctr_accessor common_table)

cc_library(table SRCS table.cc DEPS memory_sparse_table common_table tensor_accessor tensor_table ps_framework_proto string_helper device_context gflags glog boost)

target_link_libraries(table -fopenmp)

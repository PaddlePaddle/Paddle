if(APPLE)
    return()
endif()

set_source_files_properties(table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(table_test SRCS table_test.cc DEPS common_table table tensor_accessor ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(dense_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(dense_table_test SRCS dense_table_test.cc DEPS common_table table tensor_accessor ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(sparse_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(sparse_table_test SRCS sparse_table_test.cc DEPS common_table table tensor_accessor ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(geo_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(geo_table_test SRCS geo_table_test.cc DEPS common_table table tensor_accessor ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(barrier_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(barrier_table_test SRCS barrier_table_test.cc DEPS common_table table tensor_accessor ps_framework_proto ${COMMON_DEPS})
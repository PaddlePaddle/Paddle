set_source_files_properties(table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(table_test SRCS table_test.cc DEPS common_table table tensor_accessor ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(heter_serde_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(heter_serde_test SRCS heter_serde_test.cc DEPS heter_serde scope math_function ${COMMON_DEPS} ${RPC_DEPS})

set_source_files_properties(dense_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(dense_table_test SRCS dense_table_test.cc DEPS common_table table tensor_accessor ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(sparse_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(sparse_table_test SRCS sparse_table_test.cc DEPS common_table table tensor_accessor ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(geo_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(geo_table_test SRCS geo_table_test.cc DEPS common_table table tensor_accessor ps_framework_proto ${COMMON_DEPS})

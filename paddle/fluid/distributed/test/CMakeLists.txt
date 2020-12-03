set_source_files_properties(table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(table_test SRCS table_test.cc DEPS common_table table tensor_accessor ps_framework_proto ${COMMON_DEPS})

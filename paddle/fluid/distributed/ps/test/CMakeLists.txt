set_source_files_properties(feature_value_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(feature_value_test SRCS feature_value_test.cc DEPS ${COMMON_DEPS} boost table)

set_source_files_properties(feature_value_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(feature_value_test SRCS feature_value_test.cc DEPS ${COMMON_DEPS} boost table)


set_source_files_properties(sparse_sgd_rule_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(sparse_sgd_rule_test SRCS sparse_sgd_rule_test.cc DEPS ${COMMON_DEPS} boost table)

set_source_files_properties(ctr_accessor_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(ctr_accessor_test SRCS ctr_accessor_test.cc DEPS ${COMMON_DEPS} boost table)

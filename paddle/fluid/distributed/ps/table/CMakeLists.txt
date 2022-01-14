set_source_files_properties(sparse_sgd_rule.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ctr_accessor.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_library(sparse_sgd_rule SRCS sparse_sgd_rule.cc DEPS ${TABLE_DEPS} ps_framework_proto)
cc_library(ctr_accessor SRCS ctr_accessor.cc DEPS ${TABLE_DEPS} ps_framework_proto sparse_sgd_rule)

set_source_files_properties(table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(table_test SRCS table_test.cc DEPS common_table table tensor_accessor ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(brpc_service_dense_sgd_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(brpc_service_dense_sgd_test SRCS brpc_service_dense_sgd_test.cc DEPS scope server client communicator ps_service boost table ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(brpc_service_sparse_sgd_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(brpc_service_sparse_sgd_test SRCS brpc_service_sparse_sgd_test.cc DEPS scope server client communicator ps_service boost table ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(heter_serde_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(heter_serde_test SRCS heter_serde_test.cc DEPS heter_serde scope math_function ${COMMON_DEPS} ${RPC_DEPS})
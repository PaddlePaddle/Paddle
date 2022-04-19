if (WITH_PSLIB)
    return()
endif()

include(operators)

set(DISTRIBUTE_DEPS "")

if (WITH_ARM_BRPC)
    list(APPEND DISTRIBUTE_DEPS executor fleet ps_service brpc_utils heter_server heter_client ps_framework_proto framework_proto sendrecv_rpc arm_brpc gflags glog snappy device_context)
else()
    list(APPEND DISTRIBUTE_DEPS executor fleet ps_service brpc_utils heter_server heter_client ps_framework_proto framework_proto sendrecv_rpc brpc leveldb ssl crypto protobuf gflags glog zlib snappy device_context)
endif()

set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=parentheses")

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(DISTRIBUTE_COMPILE_FLAGS
            "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()

file(GLOB OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_op.cc")
list(REMOVE_DUPLICATES OPS)

foreach (src ${OPS})
    set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
endforeach ()

register_operators(DEPS ${DISTRIBUTE_DEPS})

set(OPERATOR_DEPS ${OPERATOR_DEPS} ${DISTRIBUTE_DEPS} PARENT_SCOPE)

set_source_files_properties(heter_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(heter_server_test SRCS heter_server_test.cc DEPS ${RPC_DEPS} ${DISTRIBUTE_DEPS} executor scope proto_desc scale_op eigen_function)

set_source_files_properties(send_and_recv_op_cpu_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(send_and_recv_cpu_test SRCS send_and_recv_op_cpu_test.cc DEPS executor scope proto_desc scale_op send_and_recv_op ${RPC_DEPS} ${DISTRIBUTE_DEPS} eigen_function)

set_source_files_properties(send_and_recv_op_gpu_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(send_and_recv_gpu_test SRCS send_and_recv_op_gpu_test.cc DEPS executor scope proto_desc scale_op send_and_recv_op ${RPC_DEPS} ${DISTRIBUTE_DEPS} eigen_function)

set_source_files_properties(heter_listen_and_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(heter_listen_and_server_test SRCS heter_listen_and_server_test.cc DEPS executor scope proto_desc scale_op heter_listen_and_serv_op ${RPC_DEPS} ${DISTRIBUTE_DEPS} eigen_function)

#set_source_files_properties(heter_cloud_comm_cpu_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
#cc_test(heter_cloud_comm_cpu_test SRCS heter_cloud_comm_cpu_test.cc DEPS executor scope proto_desc scale_op heter_listen_and_serv_op ${RPC_DEPS} ${DISTRIBUTE_DEPS} eigen_function)

set_source_files_properties(switch_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_binary(switch_server_test SRCS switch_server_test.cc DEPS executor scope proto_desc scale_op heter_listen_and_serv_op ${RPC_DEPS} ${DISTRIBUTE_DEPS} eigen_function)

if(WITH_PSLIB)
  return()
endif()

set(DISTRIBUTE_DEPS "")

if(WITH_ARM_BRPC)
  list(
    APPEND
    DISTRIBUTE_DEPS
    executor
    fleet
    ps_service
    brpc_utils
    heter_server
    heter_client
    ps_framework_proto
    framework_proto
    sendrecv_rpc
    arm_brpc
    common
    glog
    snappy
    device_context)
else()
  list(
    APPEND
    DISTRIBUTE_DEPS
    executor
    fleet
    ps_service
    brpc_utils
    heter_server
    heter_client
    ps_framework_proto
    framework_proto
    sendrecv_rpc
    ${EXTERNAL_BRPC_DEPS}
    zlib
    device_context)
endif()

set(DISTRIBUTE_COMPILE_FLAGS
    "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=parentheses"
)

if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()

set_source_files_properties(
  heter_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(heter_server_test SRCS heter_server_test.cc)

set_source_files_properties(
  send_and_recv_op_cpu_test.cc PROPERTIES COMPILE_FLAGS
                                          ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(send_and_recv_cpu_test SRCS send_and_recv_op_cpu_test.cc)

set_source_files_properties(
  send_and_recv_op_gpu_test.cc PROPERTIES COMPILE_FLAGS
                                          ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(send_and_recv_gpu_test SRCS send_and_recv_op_gpu_test.cc)

set_source_files_properties(
  heter_listen_and_server_test.cc PROPERTIES COMPILE_FLAGS
                                             ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(heter_listen_and_server_test SRCS heter_listen_and_server_test.cc)

#set_source_files_properties(heter_cloud_comm_cpu_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
#cc_test(heter_cloud_comm_cpu_test SRCS heter_cloud_comm_cpu_test.cc DEPS executor scope proto_desc generated_static_op heter_listen_and_serv_op ${RPC_DEPS} ${DISTRIBUTE_DEPS} phi common)

set_source_files_properties(
  switch_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
paddle_test(switch_server_test SRCS switch_server_test.cc)

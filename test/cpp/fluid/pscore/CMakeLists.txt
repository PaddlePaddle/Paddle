if(WITH_PSLIB)
  return()
endif()

set(DISTRIBUTE_COMPILE_FLAGS
    "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=parentheses"
)

set_source_files_properties(
  heter_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test_old(
  heter_server_test
  SRCS
  heter_server_test.cc
  DEPS
  ${RPC_DEPS}
  ${DISTRIBUTE_DEPS}
  executor
  scope
  proto_desc
  generated_op
  eigen_function)

set_source_files_properties(
  send_and_recv_op_cpu_test.cc PROPERTIES COMPILE_FLAGS
                                          ${DISTRIBUTE_COMPILE_FLAGS})
cc_test_old(
  send_and_recv_cpu_test
  SRCS
  send_and_recv_op_cpu_test.cc
  DEPS
  executor
  scope
  proto_desc
  generated_op
  send_and_recv_op
  ${RPC_DEPS}
  ${DISTRIBUTE_DEPS}
  eigen_function)

set_source_files_properties(
  send_and_recv_op_gpu_test.cc PROPERTIES COMPILE_FLAGS
                                          ${DISTRIBUTE_COMPILE_FLAGS})
cc_test_old(
  send_and_recv_gpu_test
  SRCS
  send_and_recv_op_gpu_test.cc
  DEPS
  executor
  scope
  proto_desc
  generated_op
  send_and_recv_op
  ${RPC_DEPS}
  ${DISTRIBUTE_DEPS}
  eigen_function)

set_source_files_properties(
  heter_listen_and_server_test.cc PROPERTIES COMPILE_FLAGS
                                             ${DISTRIBUTE_COMPILE_FLAGS})
cc_test_old(
  heter_listen_and_server_test
  SRCS
  heter_listen_and_server_test.cc
  DEPS
  executor
  scope
  proto_desc
  generated_op
  heter_listen_and_serv_op
  ${RPC_DEPS}
  ${DISTRIBUTE_DEPS}
  eigen_function)

#set_source_files_properties(heter_cloud_comm_cpu_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
#cc_test(heter_cloud_comm_cpu_test SRCS heter_cloud_comm_cpu_test.cc DEPS executor scope proto_desc generated_static_op heter_listen_and_serv_op ${RPC_DEPS} ${DISTRIBUTE_DEPS} eigen_function)

set_source_files_properties(
  switch_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_binary(
  switch_server_test
  SRCS
  switch_server_test.cc
  DEPS
  executor
  scope
  proto_desc
  generated_op
  heter_listen_and_serv_op
  ${RPC_DEPS}
  ${DISTRIBUTE_DEPS}
  eigen_function)

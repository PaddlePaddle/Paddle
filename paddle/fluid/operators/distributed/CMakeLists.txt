if(NOT WITH_DISTRIBUTE)
    return()
endif()

if(WITH_GRPC)
    set(cc_generic_services "false")
else()
    set(cc_generic_services "true")
endif()
configure_file(send_recv.proto.in ${CMAKE_CURRENT_BINARY_DIR}/send_recv.proto @ONLY)

# FIXME(typhoonzero): use add_subdirectory once we clean the dependency of these files
set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
if(WITH_GRPC)
  set(GRPC_SRCS grpc/grpc_client.cc grpc/grpc_server.cc grpc/grpc_serde.cc grpc/grpc_bytebuffer_stream.cc grpc/grpc_variable_response.cc)
  grpc_library(sendrecvop_rpc SRCS sendrecvop_utils.cc
        request_handler_impl.cc rpc_client.cc rpc_server.cc
        variable_response.cc
        collective_client.cc collective_server.cc
        ${GRPC_SRCS}
      PROTO ${CMAKE_CURRENT_BINARY_DIR}/send_recv.proto 
      DEPS lod_tensor selected_rows_functor memory)

  set_source_files_properties(grpc_serde_test.cc rpc_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  set(RPC_DEPS sendrecvop_rpc grpc++_unsecure grpc_unsecure gpr cares zlib protobuf)

  cc_test(grpc_serde_test SRCS grpc/grpc_serde_test.cc 
    DEPS ${RPC_DEPS} scope profiler math_function SERIAL)

else()
  set_source_files_properties(brpc_server.cc parameter_prefetch.cc brpc_client.cc rpc_server_test.cc brpc_serde_test.cc
      brpc_variable_response.cc brpc_sendrecvop_utils.cc brpc_rdma_pool.cc collective_server.cc collective_server_test.cc
      collective_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

  set(BRPC_SRCS brpc/brpc_client.cc brpc/brpc/server.cc brpc/brpc_sendrecvop_utils.cc brpc/brpc_variable_response.cc brpc/brpc_rdma_pool.cc)
  brpc_library(sendrecvop_rpc SRCS sendrecvop_utils.cc
      request_handler_impl.cc rpc_client.cc rpc_server.cc
      variable_response.cc
      collective_client.cc collective_server.cc
      ${BRPC_SRCS}
    PROTO ${CMAKE_CURRENT_BINARY_DIR}/send_recv.proto
    DEPS lod_tensor selected_rows memory)

  set(RPC_DEPS sendrecvop_rpc brpc ssl crypto protobuf leveldb snappystream snappy zlib)
  cc_test(brpc_serde_test SRCS brpc/brpc_serde_test.cc
      DEPS ${RPC_DEPS} gflags glog executor proto_desc lookup_sparse_table_op SERIAL)
endif()


cc_test(rpc_server_test SRCS rpc_server_test.cc
    DEPS ${RPC_DEPS} executor proto_desc lookup_sparse_table_op SERIAL)
cc_test(varhandle_test SRCS varhandle_test.cc DEPS profiler)
cc_library(parameter_prefetch SRCS parameter_prefetch.cc DEPS sendrecvop_rpc memory)
if(WITH_GPU)
    cc_test(collective_server_test SRCS collective_server_test.cc 
        DEPS sendrecvop_rpc grpc++_unsecure grpc_unsecure gpr cares zlib protobuf executor
        selected_rows_functor  scope math_function SERIAL)
endif()

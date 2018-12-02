if(NOT WITH_DISTRIBUTE)
    return()
endif()

if(WITH_GRPC)
    set(cc_generic_services "false")
else()
    set(cc_generic_services "true")
endif()
configure_file(send_recv.proto.in ${CMAKE_CURRENT_SOURCE_DIR}/send_recv.proto @ONLY)

set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")

if(WITH_GRPC)
  grpc_library(sendrecvop_grpc SRCS grpc_bytebuffer_stream.cc sendrecvop_utils.cc grpc_client.cc
        request_handler_impl.cc rpc_client.cc rpc_server.cc grpc_server.cc variable_response.cc grpc_variable_response.cc grpc_serde.cc
      PROTO send_recv.proto 
      DEPS lod_tensor selected_rows memory)

  set_source_files_properties(grpc_serde_test.cc rpc_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  cc_test(grpc_serde_test SRCS grpc_serde_test.cc 
    DEPS grpc++_unsecure grpc_unsecure gpr cares zlib protobuf sendrecvop_grpc scope profiler math_function SERIAL)
  cc_test(rpc_server_test SRCS rpc_server_test.cc
    DEPS sendrecvop_grpc grpc++_unsecure grpc_unsecure gpr cares zlib protobuf executor  proto_desc lookup_sparse_table_op SERIAL)
  cc_test(varhandle_test SRCS varhandle_test.cc DEPS profiler)
  cc_library(parameter_prefetch SRCS parameter_prefetch.cc DEPS sendrecvop_grpc memory)
else()
  set_source_files_properties(brpc_server.cc brpc_client.cc rpc_server_test.cc brpc_serde_test.cc
      brpc_variable_response.cc brpc_sendrecvop_utils.cc brpc_rdma_pool.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

  brpc_library(sendrecvop_brpc SRCS brpc_client.cc brpc_server.cc rpc_server.cc rpc_client.cc request_handler_impl.cc brpc_sendrecvop_utils.cc
      brpc_variable_response.cc variable_response.cc sendrecvop_utils.cc brpc_rdma_pool.cc
    PROTO send_recv.proto
    DEPS lod_tensor selected_rows memory)

  cc_library(parameter_prefetch SRCS parameter_prefetch.cc DEPS sendrecvop_brpc memory)

  set(brpc_test_depends sendrecvop_brpc brpc ssl crypto protobuf leveldb gflags glog executor proto_desc lookup_table_op snappystream snappy)

  cc_test(brpc_server_test SRCS rpc_server_test.cc
      DEPS ${brpc_test_depends} SERIAL)

  cc_test(brpc_serde_test SRCS brpc_serde_test.cc
      DEPS ${brpc_test_depends} SERIAL)
endif()

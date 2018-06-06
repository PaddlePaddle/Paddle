if(NOT WITH_DISTRIBUTE)
    return()
endif()


if(WITH_GRPC)
  grpc_library(sendrecvop_grpc SRCS bytebuffer_stream.cc sendrecvop_utils.cc grpc_client.cc
      request_handler_impl.cc rpc_server.cc grpc_server.cc variable_response.cc PROTO send_recv.proto DEPS lod_tensor
      selected_rows memory)
  set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
  set_source_files_properties(serde_test.cc grpc_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  cc_test(serde_test SRCS serde_test.cc variable_response.cc DEPS grpc++_unsecure grpc_unsecure gpr
          cares zlib protobuf sendrecvop_grpc SERIAL)
  cc_test(grpc_server_test SRCS grpc_server_test.cc DEPS sendrecvop_grpc
          grpc++_unsecure grpc_unsecure gpr cares zlib protobuf executor
          proto_desc lookup_table_op SERIAL)
endif()


if(WITH_BRPC)
  brpc_library(sendrecvop_brpc SRCS brpc_client.cc brpc_server.cc request_handler_impl.cc rpc_server.cc 
      PROTO send_recv.proto 
      DEPS lod_tensor selected_rows memory)

  # cc_test(serde_test SRCS serde_test.cc  DEPS brpc sendrecvop_grpc)
  # cc_test(brpc_server_test SRCS brpc_server_test.cc DEPS sendrecvop_brpc protobuf executor
  #        proto_desc lookup_table_op)
endif()

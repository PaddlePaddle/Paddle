if(WITH_DISTRIBUTE)
  grpc_library(sendrecvop_grpc SRCS bytebuffer_stream.cc sendrecvop_utils.cc grpc_client.cc
      grpc_server.cc variable_response.cc PROTO send_recv.proto DEPS lod_tensor selected_rows)
  set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
  set_source_files_properties(serde_test.cc grpc_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  cc_test(serde_test SRCS serde_test.cc variable_response.cc DEPS grpc++_unsecure grpc_unsecure gpr
          cares zlib protobuf sendrecvop_grpc SERIAL)
  cc_test(grpc_server_test SRCS grpc_server_test.cc DEPS sendrecvop_grpc
          grpc++_unsecure grpc_unsecure gpr cares zlib protobuf executor
          proto_desc lookup_table_op SERIAL)
endif()

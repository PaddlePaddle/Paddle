grpc_library(sendrecvop_rpc SRCS grpc_bytebuffer_stream.cc grpc_client.cc
    grpc_server.cc grpc_variable_response.cc grpc_serde.cc
    PROTO ${CMAKE_CURRENT_BINARY_DIR}/../send_recv.proto 
    DEPS distributed_base lod_tensor selected_rows_functor memory)

set_source_files_properties(grpc_serde_test.cc rpc_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

set(GRPC_DEPS grpc++_unsecure grpc_unsecure gpr cares zlib protobuf)
set(RPC_DEPS ${GRPC_DEPS} PARENT_SCOPE)

cc_test(grpc_serde_test SRCS grpc_serde_test.cc 
  DEPS ${GRPC_DEPS} sendrecvop_rpc scope profiler math_function SERIAL)

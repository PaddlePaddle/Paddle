grpc_library(sendrecvop_grpc SRCS recv_impl.cc send_impl.cc grpc_client.cc grpc_server.cc PROTO send_recv.proto DEPS lod_tensor selected_rows)

set_source_files_properties(
    grpc_test.cc
    PROPERTIES
    COMPILE_FLAGS  "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")

set(DISTRIBUTE_DEPS sendrecvop_grpc grpc++_unsecure grpc_unsecure gpr cares zlib_target protobuf)
cc_test(grpc_test SRCS grpc_test.cc DEPS ${DISTRIBUTE_DEPS})

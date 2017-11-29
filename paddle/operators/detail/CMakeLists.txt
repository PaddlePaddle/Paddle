grpc_library(sendrecvop_grpc SRCS recv_impl.cc send_impl.cc PROTO send_recv.proto DEPS lod_tensor selected_rows)
set_source_files_properties(
    recv_impl.cc send_impl.cc send_recv.grpc.pb.cc
    PROPERTIES
    COMPILE_FLAGS  "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")

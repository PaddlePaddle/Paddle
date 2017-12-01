grpc_library(sendrecvop_grpc SRCS recv_impl.cc send_impl.cc PROTO send_recv.proto DEPS lod_tensor selected_rows)

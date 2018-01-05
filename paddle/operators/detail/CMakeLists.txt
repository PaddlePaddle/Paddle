grpc_library(sendrecvop_grpc SRCS grpc_server.cc PROTO send_recv.proto DEPS lod_tensor selected_rows)

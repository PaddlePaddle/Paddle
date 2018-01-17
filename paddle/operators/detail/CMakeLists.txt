grpc_library(sendrecvop_grpc SRCS sendrecvop_utils.cc grpc_client.cc grpc_server.cc PROTO send_recv.proto DEPS lod_tensor selected_rows)

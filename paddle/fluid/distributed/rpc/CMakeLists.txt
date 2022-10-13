set(PADDLE_RPC_SRCS python_rpc_handler.cc rpc_agent.cc)

set_source_files_properties(
  python_rpc_handler.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(rpc_agent.cc PROPERTIES COMPILE_FLAGS
                                                    ${DISTRIBUTE_COMPILE_FLAGS})

set(PADDLE_RPC_DEPS brpc protobuf glog pybind)
proto_library(paddle_rpc_proto SRCS rpc.proto)
cc_library(
  paddle_rpc
  SRCS ${PADDLE_RPC_SRCS}
  DEPS ${PADDLE_RPC_DEPS} paddle_rpc_proto)

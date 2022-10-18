set(PADDLE_RPC_SRCS python_rpc_handler.cc rpc_agent.cc)
set(DISTRIBUTE_COMPILE_FLAGS
    "-Wno-error=unused-value -Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=sign-compare -Wno-error=unused-variable -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=unknown-pragmas -Wno-error=parentheses -Wno-error=unused-result"
)

if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()

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
add_dependencies(paddle_rpc brpc)

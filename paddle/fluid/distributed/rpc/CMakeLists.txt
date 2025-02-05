set(PADDLE_RPC_SRCS python_rpc_handler.cc rpc_agent.cc)
set(DISTRIBUTE_COMPILE_FLAGS
    "-Wno-error=unused-value -Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=parentheses -Wno-error=unused-result"
)

if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()

set_source_files_properties(
  python_rpc_handler.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(rpc_agent.cc PROPERTIES COMPILE_FLAGS
                                                    ${DISTRIBUTE_COMPILE_FLAGS})

set(PADDLE_RPC_DEPS ${EXTERNAL_BRPC_DEPS} zlib phi common pybind)
proto_library(paddle_rpc_proto SRCS rpc.proto)
cc_library(
  paddle_rpc
  SRCS ${PADDLE_RPC_SRCS}
  DEPS ${PADDLE_RPC_DEPS} paddle_rpc_proto pybind)
add_dependencies(paddle_rpc brpc)

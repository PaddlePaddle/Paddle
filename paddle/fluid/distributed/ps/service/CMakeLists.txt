set(BRPC_SRCS ps_client/ps_client.cc ps_server/server.cc)
set_source_files_properties(${BRPC_SRCS})

set(BRPC_DEPS brpc ssl crypto protobuf gflags glog zlib leveldb snappy gflags glog device_context)

brpc_library(sendrecv_rpc SRCS
        ${BRPC_SRCS}
        PROTO sendrecv.proto
        DEPS ${BRPC_DEPS} )

#set_property(GLOBAL PROPERTY RPC_DEPS sendrecv_rpc ${BRPC_DEPS} string_helper)

get_property(RPC_DEPS GLOBAL PROPERTY RPC_DEPS)

set_source_files_properties(communicator/communicator.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ps_service/service.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ps_server/brpc_ps_server.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ps_client/brpc_ps_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ps_client/ps_local_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

set_source_files_properties(brpc_utils.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ps_server/heter_server.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ps_client/heter_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

set_source_files_properties(ps_client/client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ps_client/ps_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ps_server/server.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ps_server/graph_brpc_server.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ps_client/graph_brpc_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_library(brpc_utils SRCS brpc_utils.cc DEPS tensor device_context ${COMMON_DEPS} ${RPC_DEPS})

cc_library(downpour_server SRCS ps_server/graph_brpc_server.cc ps_server/brpc_ps_server.cc DEPS boost eigen3 table brpc_utils simple_threadpool ${RPC_DEPS})
cc_library(downpour_client SRCS ps_client/graph_brpc_client.cc ps_client/brpc_ps_client.cc
ps_client/ps_local_client.cc DEPS boost eigen3 table brpc_utils simple_threadpool ${RPC_DEPS})

cc_library(client SRCS ps_client/ps_client.cc DEPS downpour_client boost ${RPC_DEPS})
cc_library(server SRCS ps_server/server.cc DEPS downpour_server boost ${RPC_DEPS})

cc_library(communicator SRCS communicator/communicator.cc DEPS scope client boost table math_function selected_rows_functor ${RPC_DEPS})
cc_library(ps_service SRCS ps_service/service.cc DEPS communicator client server boost ${RPC_DEPS})

cc_library(heter_server SRCS ps_server/heter_server.cc DEPS brpc_utils ${COMMON_DEPS} ${RPC_DEPS})
cc_library(heter_client SRCS ps_client/heter_client.cc DEPS brpc_utils ${COMMON_DEPS} ${RPC_DEPS})

set_source_files_properties(ps_service/graph_py_service.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_library(graph_py_service SRCS ps_service/graph_py_service.cc DEPS ps_service)

#add_subdirectory(communicator)

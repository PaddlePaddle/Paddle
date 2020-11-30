
set(BRPC_SRCS ps_client.cc server.cc)
set_source_files_properties(${BRPC_SRCS})

set(BRPC_DEPS brpc ssl crypto protobuf gflags glog zlib leveldb snappy gflags glog)

brpc_library(sendrecv_rpc SRCS
        ${BRPC_SRCS}
        PROTO sendrecv.proto
        DEPS ${BRPC_DEPS} )

set_property(GLOBAL PROPERTY RPC_DEPS sendrecv_rpc ${BRPC_DEPS})

get_property(RPC_DEPS GLOBAL PROPERTY RPC_DEPS)

set_source_files_properties(communicator.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(service.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(brpc_ps_server.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(brpc_ps_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

set_source_files_properties(ps_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(server.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

cc_library(downpour_server SRCS brpc_ps_server.cc DEPS boost eigen3 table ${RPC_DEPS})
cc_library(downpour_client SRCS brpc_ps_client.cc DEPS boost eigen3 table ${RPC_DEPS})

cc_library(client SRCS ps_client.cc DEPS downpour_client boost ${RPC_DEPS})
cc_library(server SRCS server.cc DEPS downpour_server boost ${RPC_DEPS})

cc_library(communicator SRCS communicator.cc DEPS scope client boost table ${RPC_DEPS})
cc_library(ps_service SRCS service.cc DEPS communicator client server boost ${RPC_DEPS})

set(BRPC_SRCS ps_client.cc server.cc)
set_source_files_properties(${BRPC_SRCS})

if(WITH_HETERPS)
    set(BRPC_DEPS brpc ssl crypto protobuf gflags glog zlib leveldb snappy gflags glog device_context rocksdb)
else()
    set(BRPC_DEPS brpc ssl crypto protobuf gflags glog zlib leveldb snappy gflags glog device_context)
endif()

brpc_library(sendrecv_rpc SRCS
        ${BRPC_SRCS}
        PROTO sendrecv.proto
        DEPS ${BRPC_DEPS} )

#set_property(GLOBAL PROPERTY RPC_DEPS sendrecv_rpc ${BRPC_DEPS} string_helper)

get_property(RPC_DEPS GLOBAL PROPERTY RPC_DEPS)

set_source_files_properties(communicator/communicator.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ps_service/service.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(brpc_ps_server.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(brpc_ps_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ps_local_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

set_source_files_properties(brpc_utils.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(heter_server.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(heter_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

set_source_files_properties(client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(ps_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(server.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(graph_brpc_server.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(graph_brpc_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_library(brpc_utils SRCS brpc_utils.cc DEPS tensor device_context ${COMMON_DEPS} ${RPC_DEPS})

cc_library(downpour_server SRCS graph_brpc_server.cc brpc_ps_server.cc DEPS boost eigen3 table brpc_utils simple_threadpool ${RPC_DEPS})
cc_library(downpour_client SRCS graph_brpc_client.cc brpc_ps_client.cc
ps_local_client.cc DEPS boost eigen3 table brpc_utils simple_threadpool ${RPC_DEPS})

cc_library(client SRCS ps_client.cc DEPS downpour_client boost ${RPC_DEPS})
cc_library(server SRCS server.cc DEPS downpour_server boost ${RPC_DEPS})

cc_library(communicator SRCS communicator/communicator.cc DEPS scope client boost table math_function selected_rows_functor ${RPC_DEPS})
cc_library(ps_service SRCS ps_service/service.cc DEPS communicator client server boost ${RPC_DEPS})

cc_library(heter_client SRCS heter_client.cc DEPS brpc_utils ${COMMON_DEPS} ${RPC_DEPS})
cc_library(heter_server SRCS heter_server.cc DEPS heter_client brpc_utils ${COMMON_DEPS} ${RPC_DEPS})

set_source_files_properties(ps_service/graph_py_service.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_library(graph_py_service SRCS ps_service/graph_py_service.cc DEPS ps_service)

#add_subdirectory(communicator)

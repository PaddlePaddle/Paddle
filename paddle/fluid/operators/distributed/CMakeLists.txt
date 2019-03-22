if(NOT WITH_DISTRIBUTE)
    return()
endif()

if(WITH_GRPC)
    set(cc_generic_services "false")
else()
    set(cc_generic_services "true")
endif()
configure_file(send_recv.proto.in ${CMAKE_CURRENT_SOURCE_DIR}/send_recv.proto @ONLY)

# FIXME(typhoonzero): use add_subdirectory once we clean the dependency of these files
set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
if(WITH_GRPC)
  set(GRPC_DEPS grpc++_unsecure grpc_unsecure gpr cares zlib protobuf)
  set(GRPC_SRCS grpc/grpc_client.cc grpc/grpc_server.cc grpc/grpc_serde.cc grpc/grpc_bytebuffer_stream.cc grpc/grpc_variable_response.cc)
  grpc_library(sendrecvop_rpc SRCS sendrecvop_utils.cc
        request_handler_impl.cc rpc_client.cc rpc_server.cc
        variable_response.cc
        collective_client.cc collective_server.cc
        ${GRPC_SRCS}
      PROTO send_recv.proto 
      DEPS lod_tensor selected_rows_functor memory scope ${GRPC_DEPS})

  set_source_files_properties(grpc_serde_test.cc rpc_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  set(RPC_DEPS sendrecvop_rpc ${GRPC_DEPS})

  cc_test(grpc_serde_test SRCS grpc/grpc_serde_test.cc 
    DEPS ${RPC_DEPS} scope profiler math_function SERIAL)

else()
  set(BRPC_SRCS brpc/brpc_client.cc brpc/brpc_server.cc brpc/brpc_sendrecvop_utils.cc brpc/brpc_variable_response.cc brpc/brpc_rdma_pool.cc)
  set_source_files_properties(${BRPC_SRCS} parameter_prefetch.cc parameter_send.cc parameter_recv.cc communicator.cc rpc_server_test.cc brpc/brpc_serde_test.cc collective_server.cc collective_server_test.cc collective_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

  set(BRPC_DEPS brpc ssl crypto protobuf leveldb snappystream snappy zlib)

  brpc_library(sendrecvop_rpc SRCS sendrecvop_utils.cc
      request_handler_impl.cc rpc_client.cc rpc_server.cc
      variable_response.cc
      collective_client.cc collective_server.cc
      ${BRPC_SRCS}
    PROTO send_recv.proto
    DEPS lod_tensor selected_rows memory scope ${BRPC_DEPS})

  set(RPC_DEPS sendrecvop_rpc ${BRPC_DEPS})
  cc_test(brpc_serde_test SRCS brpc/brpc_serde_test.cc
      DEPS ${RPC_DEPS} gflags glog executor proto_desc lookup_sparse_table_op SERIAL)
endif()


cc_test(rpc_server_test SRCS rpc_server_test.cc
    DEPS ${RPC_DEPS} executor proto_desc lookup_sparse_table_op SERIAL)
cc_test(varhandle_test SRCS varhandle_test.cc DEPS profiler scope)
cc_test(async_sparse_param_update_recorder_test SRCS async_sparse_param_update_recorder_test.cc DEPS simple_threadpool)
cc_library(parameter_prefetch SRCS parameter_prefetch.cc DEPS sendrecvop_rpc memory)
cc_library(parameter_send SRCS parameter_send.cc DEPS sendrecvop_rpc memory)
cc_library(parameter_recv SRCS parameter_recv.cc DEPS sendrecvop_rpc memory)
cc_library(communicator SRCS communicator.cc DEPS scope selected_rows tensor variable_helper selected_rows_functor simple_threadpool parameter_send parameter_recv)
cc_test(communicator_test SRCS communicator_test.cc DEPS communicator)
if(WITH_GPU)
    cc_test(collective_server_test SRCS collective_server_test.cc 
        DEPS sendrecvop_rpc executor ${RPC_DEPS}
        selected_rows_functor  scope math_function SERIAL)
endif()

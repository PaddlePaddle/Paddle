if(NOT WITH_DISTRIBUTE)
    return()
endif()

if(WITH_GRPC)
    set(cc_generic_services "false")
else()
    set(cc_generic_services "true")
endif()
configure_file(send_recv.proto.in ${CMAKE_CURRENT_SOURCE_DIR}/send_recv.proto @ONLY)

cc_library(barrier SRCS barrier.cc)
cc_library(rpc_server SRCS rpc_server.cc DEPS scope barrier)
cc_library(rpc_client SRCS rpc_client.cc DEPS scope device_context)

add_subdirectory(handlers)

cc_library(async_sparse_param_update_recorder SRCS async_sparse_param_update_recorder.cc DEPS enforce simple_threadpool)
cc_test(async_sparse_param_update_recorder_test SRCS async_sparse_param_update_recorder_test.cc DEPS async_sparse_param_update_recorder)

set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
if(WITH_GRPC)
  grpc_library(grpc_base PROTO send_recv.proto)
  cc_library(sendrecvop_utils SRCS sendrecvop_utils.cc DEPS data_type lod_tensor selected_rows grpc_base_grpc)
  cc_library(variable_response SRCS variable_response.cc DEPS data_type lod_tensor scope grpc_base_proto grpc_base_grpc)

  cc_library(grpc_server SRCS grpc/grpc_server.cc DEPS grpc_base_grpc scope)
  cc_library(grpc_client SRCS grpc/grpc_client.cc DEPS grpc_base_grpc scope)
  cc_library(grpc_serde SRCS grpc/grpc_serde.cc DEPS data_type grpc_base_grpc scope)
  cc_library(grpc_bbs SRCS grpc/grpc_bytebuffer_stream.cc DEPS grpc_base_grpc scope)
  cc_library(grpc_vr SRCS grpc/grpc_variable_response.cc DEPS grpc_base_grpc scope)
  cc_library(collective_server SRCS collective_server.cc DEPS grpc_base_grpc scope)
  cc_library(collective_client SRCS collective_client.cc DEPS grpc_base_grpc scope)

  # merge all
  set(GRPC_DEPS grpc++_unsecure grpc_unsecure gpr cares zlib protobuf)

  cc_library(sendrecvop_rpc DEPS grpc_base_proto grpc_base_grpc
    grpc_server grpc_client grpc_serde grpc_bbs grpc_vr
    handlers barrier rpc_server rpc_client
    collective_server collective_client
    sendrecvop_utils variable_response)


  set_source_files_properties(grpc_serde_test.cc rpc_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  set(RPC_DEPS sendrecvop_rpc ${GRPC_DEPS})

  cc_test(grpc_serde_test SRCS grpc/grpc_serde_test.cc
    DEPS ${RPC_DEPS} scope profiler math_function)
else()
  set_source_files_properties(brpc/brpc_client.cc brpc/brpc_server.cc
        brpc/brpc_sendrecvop_utils.cc brpc/brpc_variable_response.cc
        brpc/brpc_rdma_pool.cc brpc/brpc_serde_test.cc
        collective_server.cc collective_client.cc 
        parameter_prefetch.cc
        collective_server_test.cc
        rpc_server_test.cc
        varhandle_test.cc
    PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})


  set(BRPC_DEPS brpc ssl crypto protobuf leveldb snappystream snappy zlib)
  brpc_library(brpc_base PROTO send_recv.proto DEPS brpc ${BRPC_DEPS})
  cc_library(sendrecvop_utils SRCS sendrecvop_utils.cc DEPS data_type lod_tensor selected_rows brpc_base_proto)
  cc_library(variable_response SRCS variable_response.cc DEPS data_type lod_tensor scope brpc_base_proto)

  cc_library(brpc_client SRCS brpc/brpc_client.cc DEPS brpc_base_proto brpc scope)
  cc_library(brpc_server SRCS brpc/brpc_server.cc DEPS brpc_base_proto brpc scope)
  cc_library(brpc_sendrecvop_utils SRCS brpc/brpc_sendrecvop_utils.cc DEPS brpc_base_proto brpc scope)
  cc_library(brpc_variable_response SRCS brpc/brpc_variable_response.cc DEPS brpc_base_proto brpc scope)
  cc_library(brpc_rdma_pool SRC brpc/brpc_rdma_pool.cc DEPS brpc_base_proto brpc scope)

  cc_library(collective_server SRCS collective_server.cc DEPS brpc_base_proto brpc scope rpc_server brpc_server handlers)
  cc_library(collective_client SRCS collective_client.cc DEPS brpc_base_proto brpc scope rpc_client brpc_client)
  # merge all
  cc_library(sendrecvop_rpc DEPS brpc_base_proto sendrecvop_utils variable_response
             brpc_client brpc_server brpc_sendrecvop_utils
             brpc_variable_response brpc_rdma_pool
             rpc_server rpc_client handlers barrier)

  set(RPC_DEPS sendrecvop_rpc ${BRPC_DEPS})
  cc_test(brpc_serde_test SRCS brpc/brpc_serde_test.cc
      DEPS ${RPC_DEPS} gflags glog executor proto_desc lookup_sparse_table_op)
endif()


cc_test(rpc_server_test SRCS rpc_server_test.cc
    DEPS ${RPC_DEPS} executor proto_desc lookup_sparse_table_op)
cc_test(varhandle_test SRCS varhandle_test.cc DEPS profiler scope)
cc_library(parameter_prefetch SRCS parameter_prefetch.cc DEPS sendrecvop_rpc memory)
cc_library(parameter_send SRCS parameter_send.cc DEPS sendrecvop_rpc memory)
cc_library(parameter_recv SRCS parameter_recv.cc DEPS sendrecvop_rpc memory)
cc_library(communicator SRCS communicator.cc DEPS scope selected_rows tensor variable_helper selected_rows_functor simple_threadpool parameter_send parameter_recv)
cc_test(communicator_test SRCS communicator_test.cc DEPS communicator)

if(WITH_GPU)
    cc_test(collective_server_test SRCS collective_server_test.cc 
        DEPS collective_server collective_client ${RPC_DEPS}
        executor selected_rows_functor scope math_function)
endif()

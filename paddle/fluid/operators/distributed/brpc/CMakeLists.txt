# set_source_files_properties(brpc_server.cc parameter_prefetch.cc brpc_client.cc rpc_server_test.cc brpc_serde_test.cc
#     brpc_variable_response.cc brpc_sendrecvop_utils.cc brpc_rdma_pool.cc collective_server.cc collective_server_test.cc
#     collective_client.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

brpc_library(sendrecvop_rpc SRCS brpc_client.cc brpc_server.cc rpc_server.cc rpc_client.cc request_handler_impl.cc brpc_sendrecvop_utils.cc
    brpc_variable_response.cc variable_response.cc sendrecvop_utils.cc brpc_rdma_pool.cc collective_client.cc collective_server.cc
  PROTO ${CMAKE_CURRENT_BINARY_DIR}/../send_recv.proto 
  DEPS distributed_base lod_tensor selected_rows memory)

set(BRPC_DEPS brpc ssl crypto protobuf leveldb snappystream snappy zlib)
set(RPC_DEPS ${BRPC_DEPS} PARENT_SCOPE)

cc_test(brpc_serde_test SRCS brpc_serde_test.cc
    DEPS ${BRPC_DEPS} sendrecvop_rpc gflags glog executor proto_desc lookup_sparse_table_op SERIAL)

if(NOT WITH_DISTRIBUTE)
    return()
endif()

if(WITH_GRPC)
    set(cc_generic_services "true")
else()
    set(cc_generic_services "false")
endif()
configure_file(send_recv.proto.in ${CMAKE_CURRENT_SOURCE_DIR}/send_recv.proto @ONLY)

if(WITH_GRPC)
  grpc_library(sendrecvop_grpc SRCS grpc_bytebuffer_stream.cc sendrecvop_utils.cc grpc_client.cc
        request_handler_impl.cc rpc_client.cc rpc_server.cc grpc_server.cc grpc_variable_response.cc variable_response.cc
      PROTO send_recv.proto 
      DEPS lod_tensor selected_rows memory DEPENDS send_recv_proto send_recv.proto)
  set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
  set_source_files_properties(grpc_serde_test.cc rpc_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  cc_test(grpc_serde_test SRCS grpc_serde_test.cc 
     DEPS grpc++_unsecure grpc_unsecure gpr cares zlib protobuf sendrecvop_grpc SERIAL)
  cc_test(grpc_server_test SRCS rpc_server_test.cc 
    DEPS sendrecvop_grpc grpc++_unsecure grpc_unsecure gpr cares zlib protobuf executor  proto_desc lookup_table_op SERIAL)
  return()
endif()


set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
set_source_files_properties(brpc_server.cc brpc_client.cc rpc_server_test.cc brpc_serde_test.cc
    brpc_variable_response.cc brpc_sendrecvop_utils.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

brpc_library(sendrecvop_brpc SRCS brpc_client.cc brpc_server.cc rpc_server.cc rpc_client.cc request_handler_impl.cc brpc_sendrecvop_utils.cc 
    brpc_variable_response.cc variable_response.cc sendrecvop_utils.cc
  PROTO send_recv.proto
  DEPS lod_tensor selected_rows memory)

find_library(OPENSSL_CRYPTO_LIBRARY_STATIC NAMES libcrypto.so)
ADD_LIBRARY(crypto SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET crypto PROPERTY IMPORTED_LOCATION ${OPENSSL_CRYPTO_LIBRARY_STATIC})

find_library(OPENSSL_SSL_LIBRARY_STATIC NAMES libssl.so)
ADD_LIBRARY(ssl SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET ssl PROPERTY IMPORTED_LOCATION ${OPENSSL_SSL_LIBRARY_STATIC})

cc_test(brpc_server_test SRCS rpc_server_test.cc 
    DEPS sendrecvop_brpc brpc protobuf leveldb gflags glog executor proto_desc lookup_table_op snappystream snappy ssl crypto SERIAL)

cc_test(brpc_serde_test SRCS brpc_serde_test.cc 
    DEPS sendrecvop_brpc brpc protobuf leveldb gflags glog executor proto_desc lookup_table_op snappystream snappy ssl crypto SERIAL)

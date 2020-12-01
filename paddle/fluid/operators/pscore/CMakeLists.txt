include(operators)

set(DISTRIBUTE_DEPS "")

list(APPEND DISTRIBUTE_DEPS fleet ps_service heter_serde heter_server heter_client ps_framework_proto framework_proto sendrecv_rpc brpc leveldb ssl crypto protobuf gflags glog zlib snappy)

set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(DISTRIBUTE_COMPILE_FLAGS
            "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()

file(GLOB OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_op.cc")
list(REMOVE_DUPLICATES OPS)

foreach (src ${OPS})
    set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
endforeach ()

register_operators()

set(OPERATOR_DEPS ${OPERATOR_DEPS} ${DISTRIBUTE_DEPS} PARENT_SCOPE)
#set(GLOB_DISTRIBUTE_DEPS ${DISTRIBUTE_DEPS} CACHE INTERNAL "distributed dependency")

#set(DISTRIBUTE_TEST_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=sign-compare -Wno-error=unused-variable -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=type-limits -Wno-error=unknown-pragmas -Wno-error=parentheses -Wno-error=unused-result")
set_source_files_properties(heter_server_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(heter_server_test SRCS heter_server_test.cc DEPS ${RPC_DEPS} ${DISTRIBUTE_DEPS} executor scope proto_desc scale_op)

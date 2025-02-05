if(WITH_PSLIB)
  return()
endif()

include(operators)

set(DISTRIBUTE_DEPS "")

if(WITH_ARM_BRPC)
  list(
    APPEND
    DISTRIBUTE_DEPS
    executor
    fleet
    ps_service
    brpc_utils
    heter_server
    heter_client
    ps_framework_proto
    sendrecv_rpc
    arm_brpc
    phi
    common
    glog
    snappy)
else()
  list(
    APPEND
    DISTRIBUTE_DEPS
    executor
    fleet
    ps_service
    brpc_utils
    heter_server
    heter_client
    ps_framework_proto
    sendrecv_rpc
    ${EXTERNAL_BRPC_DEPS}
    phi
    common
    zlib)
endif()

set(DISTRIBUTE_COMPILE_FLAGS
    "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=parentheses"
)

if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()

file(
  GLOB OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*_op.cc")
list(REMOVE_DUPLICATES OPS)

foreach(src ${OPS})
  set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS
                                                ${DISTRIBUTE_COMPILE_FLAGS})
endforeach()

register_operators(DEPS ${DISTRIBUTE_DEPS})

set(OPERATOR_DEPS
    ${OPERATOR_DEPS} ${DISTRIBUTE_DEPS}
    PARENT_SCOPE)

if (NOT WITH_ANAKIN)
  return()
endif()

INCLUDE(ExternalProject)
set(ANAKIN_SOURCE_DIR  ${THIRD_PARTY_PATH}/anakin)
# the anakin install dir is only default one now
set(ANAKIN_INSTALL_DIR ${THIRD_PARTY_PATH}/anakin/src/extern_anakin/output)
set(ANAKIN_INCLUDE     ${ANAKIN_INSTALL_DIR})
set(ANAKIN_LIBRARY     ${ANAKIN_INSTALL_DIR})
set(ANAKIN_SHARED_LIB  ${ANAKIN_LIBRARY}/libanakin.so)
set(ANAKIN_SABER_LIB   ${ANAKIN_LIBRARY}/libanakin_saber_common.so)

# TODO(luotao): ANAKIN_MODLE_URL will move to demo ci later.
set(ANAKIN_MODLE_URL "http://paddle-inference-dist.bj.bcebos.com/mobilenet_v2.anakin.bin")
execute_process(COMMAND bash -c "mkdir -p ${ANAKIN_SOURCE_DIR}")
execute_process(COMMAND bash -c "cd ${ANAKIN_SOURCE_DIR}; wget -q --no-check-certificate ${ANAKIN_MODLE_URL}")

include_directories(${ANAKIN_INCLUDE})
include_directories(${ANAKIN_INCLUDE}/saber/)

set(ANAKIN_COMPILE_EXTRA_FLAGS 
    -Wno-error=unused-but-set-variable -Wno-unused-but-set-variable
    -Wno-error=unused-variable -Wno-unused-variable 
    -Wno-error=format-extra-args -Wno-format-extra-args
    -Wno-error=comment -Wno-comment 
    -Wno-error=format -Wno-format 
    -Wno-error=switch -Wno-switch
    -Wno-error=return-type -Wno-return-type 
    -Wno-error=non-virtual-dtor -Wno-non-virtual-dtor
    -Wno-sign-compare
    -Wno-reorder 
    -Wno-error=cpp)

ExternalProject_Add(
    extern_anakin
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS             ${MKLML_PROJECT}
    # Anakin codes error on Intel(R) Xeon(R) Gold 5117 CPU, temporary do not compile avx512 related code.
    GIT_REPOSITORY      "https://github.com/luotao1/Anakin"
    GIT_TAG             "bcf17aabe7921ceb7bce591244b4f9dce7dba5c8"
    PREFIX              ${ANAKIN_SOURCE_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DUSE_GPU_PLACE=YES
                        -DUSE_X86_PLACE=YES
                        -DBUILD_WITH_UNIT_TEST=NO
                        -DPROTOBUF_ROOT=${THIRD_PARTY_PATH}/install/protobuf
                        -DMKLML_ROOT=${THIRD_PARTY_PATH}/install/mklml
                        -DCUDNN_ROOT=${CUDNN_ROOT}
                        ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS    -DCMAKE_INSTALL_PREFIX:PATH=${ANAKIN_INSTALL_DIR}
)

message(STATUS "Anakin for inference is enabled")
message(STATUS "Anakin is set INCLUDE:${ANAKIN_INCLUDE} LIBRARY:${ANAKIN_LIBRARY}")

add_library(anakin_shared SHARED IMPORTED GLOBAL)
set_property(TARGET anakin_shared PROPERTY IMPORTED_LOCATION ${ANAKIN_SHARED_LIB})
add_dependencies(anakin_shared extern_anakin protobuf mklml)

add_library(anakin_saber SHARED IMPORTED GLOBAL)
set_property(TARGET anakin_saber PROPERTY IMPORTED_LOCATION ${ANAKIN_SABER_LIB})
add_dependencies(anakin_saber extern_anakin protobuf mklml)

list(APPEND external_project_dependencies anakin_shared anakin_saber)

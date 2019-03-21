if (NOT WITH_ANAKIN)
  return()
endif()

option(ANAKIN_ENABLE_OP_TIMER      "Get more detailed information with Anakin op time"        OFF)
if(ANAKIN_ENABLE_OP_TIMER)
  add_definitions(-DPADDLE_ANAKIN_ENABLE_OP_TIMER)
endif()

INCLUDE(ExternalProject)
set(ANAKIN_SOURCE_DIR  ${THIRD_PARTY_PATH}/anakin)
# the anakin install dir is only default one now
set(ANAKIN_INSTALL_DIR ${THIRD_PARTY_PATH}/anakin/src/extern_anakin/output)
set(ANAKIN_INCLUDE     ${ANAKIN_INSTALL_DIR})
set(ANAKIN_LIBRARY     ${ANAKIN_INSTALL_DIR})
set(ANAKIN_SHARED_LIB  ${ANAKIN_LIBRARY}/libanakin.so)
set(ANAKIN_SABER_LIB   ${ANAKIN_LIBRARY}/libanakin_saber_common.so)

include_directories(${ANAKIN_INCLUDE})
include_directories(${ANAKIN_INCLUDE}/saber/)
include_directories(${ANAKIN_INCLUDE}/saber/core/)
include_directories(${ANAKIN_INCLUDE}/saber/funcs/impl/x86/)
include_directories(${ANAKIN_INCLUDE}/saber/funcs/impl/cuda/base/cuda_c/)

set(ANAKIN_COMPILE_EXTRA_FLAGS
    -Wno-error=unused-but-set-variable -Wno-unused-but-set-variable
    -Wno-error=unused-variable -Wno-unused-variable
    -Wno-error=format-extra-args -Wno-format-extra-args
    -Wno-error=comment -Wno-comment 
    -Wno-error=format -Wno-format 
    -Wno-error=maybe-uninitialized -Wno-maybe-uninitialized
    -Wno-error=switch -Wno-switch
    -Wno-error=return-type -Wno-return-type
    -Wno-error=non-virtual-dtor -Wno-non-virtual-dtor
    -Wno-error=ignored-qualifiers
    -Wno-ignored-qualifiers
    -Wno-sign-compare
    -Wno-reorder
    -Wno-error=cpp)

if(WITH_GPU)
    set(CMAKE_ARGS_PREFIX -DUSE_GPU_PLACE=YES -DCUDNN_ROOT=${CUDNN_ROOT} -DCUDNN_INCLUDE_DIR=${CUDNN_INCLUDE_DIR})
else()
    set(CMAKE_ARGS_PREFIX -DUSE_GPU_PLACE=NO)
endif()
ExternalProject_Add(
    extern_anakin
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS             ${MKLML_PROJECT}
    GIT_REPOSITORY      "https://github.com/PaddlePaddle/Anakin"
    GIT_TAG             "3c8554f4978628183566ab7dd6c1e7e66493c7cd"
    PREFIX              ${ANAKIN_SOURCE_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          ${CMAKE_ARGS_PREFIX}
                        -DUSE_LOGGER=YES
                        -DUSE_X86_PLACE=YES
                        -DBUILD_WITH_UNIT_TEST=NO
                        -DPROTOBUF_ROOT=${THIRD_PARTY_PATH}/install/protobuf
                        -DMKLML_ROOT=${THIRD_PARTY_PATH}/install/mklml
                        -DENABLE_OP_TIMER=${ANAKIN_ENABLE_OP_TIMER}
                        -DBUILD_FAT_BIN=${ANAKIN_BUILD_FAT_BIN}
                        -DBUILD_CROSS_PLANTFORM=${ANAKIN_BUILD_CROSS_PLANTFORM}
                        ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS    -DCMAKE_INSTALL_PREFIX:PATH=${ANAKIN_INSTALL_DIR}
)

message(STATUS "Anakin for inference is enabled")
message(STATUS "Anakin is set INCLUDE:${ANAKIN_INCLUDE} LIBRARY:${ANAKIN_LIBRARY}")
add_dependencies(extern_anakin protobuf mklml)
add_library(anakin_shared SHARED IMPORTED GLOBAL)
set_property(TARGET anakin_shared PROPERTY IMPORTED_LOCATION ${ANAKIN_SHARED_LIB})
add_dependencies(anakin_shared extern_anakin)

add_library(anakin_saber SHARED IMPORTED GLOBAL)
set_property(TARGET anakin_saber PROPERTY IMPORTED_LOCATION ${ANAKIN_SABER_LIB})
add_dependencies(anakin_saber extern_anakin)

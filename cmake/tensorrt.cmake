if(NOT WITH_GPU)
    return()
endif()

set(TENSORRT_ROOT "/usr" CACHE PATH "TENSORRT ROOT")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/include
    $ENV{TENSORRT_ROOT} $ENV{TENSORRT_ROOT}/include
    NO_DEFAULT_PATH
)

find_library(TENSORRT_LIBRARY NAMES libnvinfer.so libnvinfer.a
    PATHS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/lib
    $ENV{TENSORRT_ROOT} $ENV{TENSORRT_ROOT}/lib
    NO_DEFAULT_PATH
    DOC "Path to TensorRT library.")

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
  if(WITH_DSO)
    set(TENSORRT_FOUND ON)
  endif(WITH_DSO)
else()
    set(TENSORRT_FOUND OFF)
endif()

if(TENSORRT_FOUND)
    file(READ ${TENSORRT_INCLUDE_DIR}/NvInfer.h TENSORRT_VERSION_FILE_CONTENTS)
    string(REGEX MATCH "define NV_TENSORRT_MAJOR +([0-9]+)" TENSORRT_MAJOR_VERSION
        "${TENSORRT_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define NV_TENSORRT_MAJOR +([0-9]+)" "\\1"
        TENSORRT_MAJOR_VERSION "${TENSORRT_MAJOR_VERSION}")

    message(STATUS "Current TensorRT header is ${TENSORRT_INCLUDE_DIR}/NvInfer.h. "
        "Current TensorRT version is v${TENSORRT_MAJOR_VERSION}. ")
    include_directories(${TENSORRT_INCLUDE_DIR})
    link_directories(${TENSORRT_LIBRARY})
    add_definitions(-DPADDLE_WITH_TENSORRT)
endif()

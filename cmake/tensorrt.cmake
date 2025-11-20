if(NOT WITH_GPU OR NOT WITH_TENSORRT)
  return()
endif()

if(WIN32)
  string(REPLACE "\\" "/" TENSORRT_ROOT "${TENSORRT_ROOT}")
  string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)\\.([0-9]+)"
               TENSORRT_VERSION ${TENSORRT_ROOT})
  if(NOT TENSORRT_VERSION)
    set(MAJOR_VERSION "unknown")
  else()
    message(STATUS "TensorRT Full Version: ${TENSORRT_VERSION}")
    string(REGEX MATCH "^[0-9]+" MAJOR_VERSION ${TENSORRT_VERSION})
  endif()
  if(MAJOR_VERSION STREQUAL "10")
    message(STATUS "TensorRT version is 10, applying specific settings.")
    set(TR_INFER_LIB nvinfer_10.lib)
    set(TR_INFER_RT nvinfer_10.dll)
    set(TR_INFER_PLUGIN_RT nvinfer_plugin_10.dll)
  else()
    message(STATUS "TensorRT version is not 10, using default settings.")
    set(TR_INFER_LIB nvinfer.lib)
    set(TR_INFER_RT nvinfer.dll)
    set(TR_INFER_PLUGIN_RT nvinfer_plugin.dll)
  endif()
else()
  set(TENSORRT_ROOT
      "/usr"
      CACHE PATH "TENSORRT ROOT")
  set(TR_INFER_LIB libnvinfer.a)
  set(TR_INFER_RT libnvinfer.so)
  set(TR_INFER_PLUGIN_RT libnvinfer_plugin.so)
endif()

find_path(
  TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS ${TENSORRT_ROOT}
        ${TENSORRT_ROOT}/include
        ${TENSORRT_ROOT}/include/${CMAKE_LIBRARY_ARCHITECTURE}
        $ENV{TENSORRT_ROOT}
        $ENV{TENSORRT_ROOT}/include
        $ENV{TENSORRT_ROOT}/include/${CMAKE_LIBRARY_ARCHITECTURE}
  NO_DEFAULT_PATH)

find_path(
  TENSORRT_LIBRARY_DIR
  NAMES ${TR_INFER_LIB} ${TR_INFER_RT}
  PATHS ${TENSORRT_ROOT}
        ${TENSORRT_ROOT}/lib
        ${TENSORRT_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}
        $ENV{TENSORRT_ROOT}
        $ENV{TENSORRT_ROOT}/lib
        $ENV{TENSORRT_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}
  NO_DEFAULT_PATH
  DOC "Path to TensorRT library.")

find_library(
  TENSORRT_LIBRARY
  NAMES ${TR_INFER_LIB} ${TR_INFER_RT}
  PATHS ${TENSORRT_LIBRARY_DIR}
  NO_DEFAULT_PATH
  DOC "Path to TensorRT library.")

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
  set(TENSORRT_FOUND ON)
else()
  set(TENSORRT_FOUND OFF)
  message(
    WARNING
      "TensorRT is disabled. You are compiling PaddlePaddle with option -DWITH_TENSORRT=ON, but TensorRT is not found, please configure path to TensorRT with option -DTENSORRT_ROOT or install it."
  )
endif()

if(TENSORRT_FOUND)
  file(READ ${TENSORRT_INCLUDE_DIR}/NvInfer.h TENSORRT_VERSION_FILE_CONTENTS)
  string(REGEX MATCH "define NV_TENSORRT_MAJOR +([0-9]+)"
               TENSORRT_MAJOR_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
  string(REGEX MATCH "define NV_TENSORRT_MINOR +([0-9]+)"
               TENSORRT_MINOR_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
  string(REGEX MATCH "define NV_TENSORRT_PATCH +([0-9]+)"
               TENSORRT_PATCH_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
  string(REGEX MATCH "define NV_TENSORRT_BUILD +([0-9]+)"
               TENSORRT_BUILD_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")

  set(TRT_ENTERPRISE "OFF")
  if("${TENSORRT_MAJOR_VERSION}" STREQUAL "")
    file(READ ${TENSORRT_INCLUDE_DIR}/NvInferVersion.h
         TENSORRT_VERSION_FILE_CONTENTS)
    string(REGEX MATCH "define NV_TENSORRT_MAJOR +([0-9]+)"
                 TENSORRT_MAJOR_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
    string(REGEX MATCH "define NV_TENSORRT_MINOR +([0-9]+)"
                 TENSORRT_MINOR_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
    string(REGEX MATCH "define NV_TENSORRT_PATCH +([0-9]+)"
                 TENSORRT_PATCH_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
    string(REGEX MATCH "define NV_TENSORRT_BUILD +([0-9]+)"
                 TENSORRT_BUILD_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")

    # In TensorRT 10.12.0.36, the version macros is TRT_*_ENTERPRISE.
    if("${TENSORRT_MAJOR_VERSION}" STREQUAL "")
      string(REGEX MATCH "define TRT_MAJOR_ENTERPRISE +([0-9]+)"
                   TENSORRT_MAJOR_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
      string(REGEX MATCH "define TRT_MINOR_ENTERPRISE +([0-9]+)"
                   TENSORRT_MINOR_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
      string(REGEX MATCH "define TRT_PATCH_ENTERPRISE +([0-9]+)"
                   TENSORRT_PATCH_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
      string(REGEX MATCH "define TRT_BUILD_ENTERPRISE +([0-9]+)"
                   TENSORRT_BUILD_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
      set(TRT_ENTERPRISE "ON")
    endif()
  endif()

  if("${TENSORRT_MAJOR_VERSION}" STREQUAL "")
    message(SEND_ERROR "Failed to detect TensorRT version.")
  endif()

  string(REGEX REPLACE "define NV_TENSORRT_MAJOR +([0-9]+)" "\\1"
                       TENSORRT_MAJOR_VERSION "${TENSORRT_MAJOR_VERSION}")
  string(REGEX REPLACE "define NV_TENSORRT_MINOR +([0-9]+)" "\\1"
                       TENSORRT_MINOR_VERSION "${TENSORRT_MINOR_VERSION}")
  string(REGEX REPLACE "define NV_TENSORRT_PATCH +([0-9]+)" "\\1"
                       TENSORRT_PATCH_VERSION "${TENSORRT_PATCH_VERSION}")
  string(REGEX REPLACE "define NV_TENSORRT_BUILD +([0-9]+)" "\\1"
                       TENSORRT_BUILD_VERSION "${TENSORRT_BUILD_VERSION}")

  if("${TRT_ENTERPRISE}" STREQUAL "ON")
    string(REGEX REPLACE "define TRT_MAJOR_ENTERPRISE +([0-9]+)" "\\1"
                         TENSORRT_MAJOR_VERSION "${TENSORRT_MAJOR_VERSION}")
    string(REGEX REPLACE "define TRT_MINOR_ENTERPRISE +([0-9]+)" "\\1"
                         TENSORRT_MINOR_VERSION "${TENSORRT_MINOR_VERSION}")
    string(REGEX REPLACE "define TRT_PATCH_ENTERPRISE +([0-9]+)" "\\1"
                         TENSORRT_PATCH_VERSION "${TENSORRT_PATCH_VERSION}")
    string(REGEX REPLACE "define TRT_BUILD_ENTERPRISE +([0-9]+)" "\\1"
                         TENSORRT_BUILD_VERSION "${TENSORRT_BUILD_VERSION}")
  endif()

  message(
    STATUS
      "Current TensorRT header is ${TENSORRT_INCLUDE_DIR}/NvInfer.h. "
      "Current TensorRT version is v${TENSORRT_MAJOR_VERSION}.${TENSORRT_MINOR_VERSION}.${TENSORRT_PATCH_VERSION}.${TENSORRT_BUILD_VERSION} "
  )
  include_directories(${TENSORRT_INCLUDE_DIR})
  link_directories(${TENSORRT_LIBRARY})
  add_definitions(-DPADDLE_WITH_TENSORRT)
endif()

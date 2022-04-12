cmake_minimum_required(VERSION 3.0)
project(cpp_inference_demo CXX C)
option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   OFF)
option(USE_TENSORRT "Compile demo with TensorRT."   OFF)
option(WITH_GTEST "Compile demo with GTEST"   OFF)
option(WITH_ONNXRUNTIME       "Compile demo with ONNXRuntime"       OFF)

if(NOT WITH_STATIC_LIB)
  add_definitions("-DPADDLE_WITH_SHARED_LIB")
else()
  # PD_INFER_DECL is mainly used to set the dllimport/dllexport attribute in dynamic library mode. 
  # Set it to empty in static library mode to avoid compilation issues.
  add_definitions("/DPD_INFER_DECL=")
endif()

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()
if(NOT DEFINED DEMO_NAME)
  message(FATAL_ERROR "please set DEMO_NAME with -DDEMO_NAME=demo_name")
endif()

include_directories("${PADDLE_LIB}/")
set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/install/")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}cryptopp/include")

link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}cryptopp/lib")
link_directories("${PADDLE_LIB}/paddle/lib")
if (WITH_ONNXRUNTIME)
  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}onnxruntime/include")
  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}paddle2onnx/include")

  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}onnxruntime/lib")
  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}paddle2onnx/lib")
endif()

if (WIN32)
  add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
  option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)
  if (MSVC_STATIC_CRT)
    if (WITH_MKL)
      set(FLAG_OPENMP "/openmp")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4244 /wd4530")
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4530")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    safe_set_static_flag()
    if (WITH_STATIC_LIB)
      add_definitions(-DSTATIC_LIB)
    endif()
  endif()
else()
  if(WITH_MKL)
    set(FLAG_OPENMP "-fopenmp")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 ${FLAG_OPENMP}")
endif()

if(WITH_GPU)
  if(NOT WIN32)
    set(CUDA_LIB "/usr/local/cuda/lib64/" CACHE STRING "CUDA Library")
  else()
    set(CUDA_LIB "" CACHE STRING "CUDA_LIB")
    if("${CUDA_LIB}" STREQUAL "")
      if(DEFINED ENV{CUDA_PATH})
        set(CUDA_LIB "$ENV{CUDA_PATH}\\lib\\x64")
      else()
        set(CUDA_LIB "C:\\Program\ Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64")
      endif()
    endif()
    message(STATUS "Current CUDA lib path: ${CUDA_LIB}")
  endif(NOT WIN32)
endif()

if (USE_TENSORRT AND WITH_GPU)
  set(TENSORRT_ROOT "" CACHE STRING "The root directory of TensorRT library")
  if("${TENSORRT_ROOT}" STREQUAL "")
      message(FATAL_ERROR "The TENSORRT_ROOT is empty, you must assign it a value with CMake command. Such as: -DTENSORRT_ROOT=TENSORRT_ROOT_PATH ")
  endif()
  set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
  set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)
  file(READ ${TENSORRT_INCLUDE_DIR}/NvInfer.h TENSORRT_VERSION_FILE_CONTENTS)
  string(REGEX MATCH "define NV_TENSORRT_MAJOR +([0-9]+)" TENSORRT_MAJOR_VERSION
    "${TENSORRT_VERSION_FILE_CONTENTS}")
  string(REGEX MATCH "define NV_TENSORRT_MINOR +([0-9]+)" TENSORRT_MINOR_VERSION
    "${TENSORRT_VERSION_FILE_CONTENTS}")
  string(REGEX MATCH "define NV_TENSORRT_PATCH +([0-9]+)" TENSORRT_PATCH_VERSION
    "${TENSORRT_VERSION_FILE_CONTENTS}")
  string(REGEX MATCH "define NV_TENSORRT_BUILD +([0-9]+)" TENSORRT_BUILD_VERSION
    "${TENSORRT_VERSION_FILE_CONTENTS}")
  if("${TENSORRT_MAJOR_VERSION}" STREQUAL "")
    file(READ ${TENSORRT_INCLUDE_DIR}/NvInferVersion.h TENSORRT_VERSION_FILE_CONTENTS)
    string(REGEX MATCH "define NV_TENSORRT_MAJOR +([0-9]+)" TENSORRT_MAJOR_VERSION
      "${TENSORRT_VERSION_FILE_CONTENTS}")
    string(REGEX MATCH "define NV_TENSORRT_MINOR +([0-9]+)" TENSORRT_MINOR_VERSION
      "${TENSORRT_VERSION_FILE_CONTENTS}")
    string(REGEX MATCH "define NV_TENSORRT_PATCH +([0-9]+)" TENSORRT_PATCH_VERSION
      "${TENSORRT_VERSION_FILE_CONTENTS}")
    string(REGEX MATCH "define NV_TENSORRT_BUILD +([0-9]+)" TENSORRT_BUILD_VERSION
      "${TENSORRT_VERSION_FILE_CONTENTS}")
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
  message(STATUS "Current TensorRT header is ${TENSORRT_INCLUDE_DIR}/NvInfer.h. "
    "Current TensorRT version is v${TENSORRT_MAJOR_VERSION}.${TENSORRT_MINOR_VERSION}.${TENSORRT_PATCH_VERSION}.${TENSORRT_BUILD_VERSION} ")
  include_directories("${TENSORRT_INCLUDE_DIR}")
  link_directories("${TENSORRT_LIB_DIR}")
  add_compile_definitions(NV_TENSORRT_MAJOR=${TENSORRT_MAJOR_VERSION})
  add_compile_definitions(NV_TENSORRT_MINOR=${TENSORRT_MINOR_VERSION})
  add_compile_definitions(NV_TENSORRT_PATCH=${TENSORRT_PATCH_VERSION})
  add_compile_definitions(NV_TENSORRT_BUILD=${TENSORRT_BUILD_VERSION})
endif()

if(WITH_MKL)
  set(MATH_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mklml")
  include_directories("${MATH_LIB_PATH}/include")
  if(WIN32)
    set(MATH_LIB ${MATH_LIB_PATH}/lib/mklml${CMAKE_STATIC_LIBRARY_SUFFIX}
                 ${MATH_LIB_PATH}/lib/libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(MATH_LIB ${MATH_LIB_PATH}/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
                 ${MATH_LIB_PATH}/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
  set(MKLDNN_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mkldnn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    if(WIN32)
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/mkldnn.lib)
    else(WIN32)
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
    endif(WIN32)
  endif()
else()
  set(OPENBLAS_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}openblas")
  include_directories("${OPENBLAS_LIB_PATH}/include/openblas")
  if(WIN32)
    set(MATH_LIB ${OPENBLAS_LIB_PATH}/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(MATH_LIB ${OPENBLAS_LIB_PATH}/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
endif()

if(WITH_STATIC_LIB)
  set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  if(WIN32)
    set(DEPS ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
endif()

if (WITH_ONNXRUNTIME)
  if(WIN32)
    set(DEPS ${DEPS} ${PADDLE_LIB_THIRD_PARTY_PATH}onnxruntime/lib/onnxruntime.lib paddle2onnx)
  elseif(APPLE)
    set(DEPS ${DEPS} ${PADDLE_LIB_THIRD_PARTY_PATH}onnxruntime/lib/libonnxruntime.1.10.0.dylib paddle2onnx)
  else()
    set(DEPS ${DEPS} ${PADDLE_LIB_THIRD_PARTY_PATH}onnxruntime/lib/libonnxruntime.so.1.10.0 paddle2onnx)
  endif()
endif()

if (NOT WIN32)
  set(EXTERNAL_LIB "-lrt -ldl -lpthread")
  set(DEPS ${DEPS}
      ${MATH_LIB} ${MKLDNN_LIB}
      glog gflags protobuf xxhash cryptopp
      ${EXTERNAL_LIB})
else()
  set(DEPS ${DEPS}
      ${MATH_LIB} ${MKLDNN_LIB}
      glog gflags_static libprotobuf xxhash cryptopp-static ${EXTERNAL_LIB})
  set(DEPS ${DEPS} shlwapi.lib)
endif(NOT WIN32)

if(WITH_GPU)
  if(NOT WIN32)
    if (USE_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    if(USE_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/nvinfer${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/nvinfer_plugin${CMAKE_STATIC_LIBRARY_SUFFIX})
      if(${TENSORRT_MAJOR_VERSION} EQUAL 7)
        set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/myelin64_1${CMAKE_STATIC_LIBRARY_SUFFIX})
      endif()
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX} )
  endif()
endif()

if(WITH_GTEST)
  include(ExternalProject)
  include(external-cmake/gtest-cpp.cmake)
endif()

add_executable(${DEMO_NAME} ${DEMO_NAME}.cc)
target_link_libraries(${DEMO_NAME} ${DEPS})
if(WITH_GTEST)
  include(GNUInstallDirs)
  include_directories(${GTEST_INSTALL_DIR}/include)
  add_dependencies(${DEMO_NAME} thirdparty_gtest)
  IF(WIN32)
    target_link_libraries(${DEMO_NAME} ${GTEST_LIBRARIES})
  ELSE()
    target_link_libraries(${DEMO_NAME} ${GTEST_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libgtest${CMAKE_STATIC_LIBRARY_SUFFIX})
  ENDIF(WIN32)
endif()
if(WIN32)
  if("${CMAKE_GENERATOR}" MATCHES "Ninja")
    set(LIB_PATH ${CMAKE_BINARY_DIR})
  else()
    set(LIB_PATH ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
  endif()

  if(USE_TENSORRT)
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_LIB_DIR}/nvinfer${CMAKE_SHARED_LIBRARY_SUFFIX}
              ${LIB_PATH}
            COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_LIB_DIR}/nvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX}
              ${LIB_PATH}
    )
    if(${TENSORRT_MAJOR_VERSION} EQUAL 7)
      add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_LIB_DIR}/myelin64_1${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${LIB_PATH})
    endif()
  endif()
  if(WITH_MKL)
    message("LIB_PATH IS ${LIB_PATH}")
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy ${MATH_LIB_PATH}/lib/mklml.dll ${LIB_PATH}
          COMMAND ${CMAKE_COMMAND} -E copy ${MATH_LIB_PATH}/lib/libiomp5md.dll ${LIB_PATH}
          COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_PATH}/lib/mkldnn.dll  ${LIB_PATH}
    )
  else()
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_LIB_PATH}/lib/openblas.dll ${LIB_PATH}
    )
  endif()
  if(WITH_ONNXRUNTIME)
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE_LIB_THIRD_PARTY_PATH}onnxruntime/lib/onnxruntime.dll
      ${LIB_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE_LIB_THIRD_PARTY_PATH}paddle2onnx/lib/paddle2onnx.dll
      ${LIB_PATH}
    )
  endif()
  if(NOT WITH_STATIC_LIB)
      add_custom_command(TARGET ${DEMO_NAME} POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_LIB}/paddle/lib/paddle_inference.dll" ${LIB_PATH}
      )
  endif()
endif()

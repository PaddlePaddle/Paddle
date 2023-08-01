if(NOT WITH_MUSA)
  return()
endif()

if(NOT DEFINED ENV{MUSA_PATH})
  set(MUSA_PATH
      "/usr/local/musa"
      CACHE PATH "Path to which ROCm has been installed")
else()
  set(MUSA_PATH
      $ENV{MUSA_PATH}
      CACHE PATH "Path to which ROCm has been installed")
endif()
set(CMAKE_MODULE_PATH "${MUSA_PATH}/cmake" ${CMAKE_MODULE_PATH})

find_package(MUSA REQUIRED)
include_directories(${MUSA_PATH}/include)
include_directories(/usr/lib/llvm-11/include/openmp/)

# TODO(@caizhi): enable finding musa version
#macro(find_musa_version version_file)
#endmacro()
#find_musa_version(${MUSA_PATH}/version.h)

if(WITH_CINN)
  list(APPEND MUSA_MCC_FLAGS -std=c++14)
else()
  list(APPEND MUSA_MCC_FLAGS -std=c++17)
endif()

list(APPEND MUSA_MCC_FLAGS --cuda-gpu-arch=mp_21)
list(APPEND MUSA_MCC_FLAGS -U__CUDA__)
#set(MUSA_VERBOSE_BUILD ON)
if(CMAKE_BUILD_TYPE MATCHES Debug)
  list(APPEND MUSA_MCC_FLAGS -g2)
  list(APPEND MUSA_MCC_FLAGS -O0)
endif()

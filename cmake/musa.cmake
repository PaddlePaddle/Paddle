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

list(APPEND MUSA_MCC_FLAGS -Wno-unknown-warning-option)
list(APPEND MUSA_MCC_FLAGS -Wno-macro-redefined)
list(APPEND MUSA_MCC_FLAGS -Wno-unused-variable)
list(APPEND MUSA_MCC_FLAGS -Wno-return-type)
list(APPEND MUSA_MCC_FLAGS -Wno-sign-compare)
list(APPEND MUSA_MCC_FLAGS -Wno-mismatched-tags)
list(APPEND MUSA_MCC_FLAGS -Wno-pessimizing-move)
list(APPEND MUSA_MCC_FLAGS -Wno-unused-but-set-variable)
list(APPEND MUSA_MCC_FLAGS -Wno-bitwise-instead-of-logical)
list(APPEND MUSA_MCC_FLAGS -Wno-format)
list(APPEND MUSA_MCC_FLAGS -Wno-unused-local-typedef)
list(APPEND MUSA_MCC_FLAGS -Wno-reorder-ctor)
list(APPEND MUSA_MCC_FLAGS -Wno-braced-scalar-init)
list(APPEND MUSA_MCC_FLAGS -Wno-pass-failed)
list(APPEND MUSA_MCC_FLAGS -Wno-missing-braces)
list(APPEND MUSA_MCC_FLAGS -Wno-dangling-gsl)

if(WITH_CINN)
  list(APPEND MUSA_MCC_FLAGS -std=c++14)
else()
  list(APPEND MUSA_MCC_FLAGS -std=c++17)
endif()

list(APPEND MUSA_MCC_FLAGS --cuda-gpu-arch=mp_21)
list(APPEND MUSA_MCC_FLAGS -U__CUDA__)
# MUSA has compile conflicts of float16.h as platform::float16 overload std::is_floating_point and std::is_integer
list(APPEND MUSA_MCC_FLAGS -D__MUSA_NO_HALF_CONVERSIONS__)

#set(MUSA_VERBOSE_BUILD ON)
if(CMAKE_BUILD_TYPE MATCHES Debug)
  list(APPEND MUSA_MCC_FLAGS -g2)
  list(APPEND MUSA_MCC_FLAGS -O0)
endif()

if(NOT WITH_ROCM)
  return()
endif()

if(NOT DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH
      "/opt/rocm"
      CACHE PATH "Path to which ROCm has been installed")
  set(HIP_PATH
      ${ROCM_PATH}/hip
      CACHE PATH "Path to which HIP has been installed")
  set(HIP_CLANG_PATH
      ${ROCM_PATH}/llvm/bin
      CACHE PATH "Path to which clang has been installed")
else()
  set(ROCM_PATH
      $ENV{ROCM_PATH}
      CACHE PATH "Path to which ROCm has been installed")
  set(HIP_PATH
      ${ROCM_PATH}/hip
      CACHE PATH "Path to which HIP has been installed")
  set(HIP_CLANG_PATH
      ${ROCM_PATH}/llvm/bin
      CACHE PATH "Path to which clang has been installed")
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
# ROCm 7 uses a unified CMake module layout under ${ROCM_PATH}/lib/cmake/hip.
if(EXISTS "${ROCM_PATH}/lib/cmake/hip")
  list(PREPEND CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip")
endif()
if(EXISTS "${ROCM_PATH}/hip/cmake")
  list(PREPEND CMAKE_MODULE_PATH "${ROCM_PATH}/hip/cmake")
endif()
set(CMAKE_PREFIX_PATH "${ROCM_PATH}" ${CMAKE_PREFIX_PATH})

find_package(HIP REQUIRED)
# hip_add_library/hip_add_executable are defined in UseHIP.cmake.
if(EXISTS "${ROCM_PATH}/lib/cmake/hip/UseHIP.cmake")
  include(UseHIP)
elseif(EXISTS "${ROCM_PATH}/hip/cmake/UseHIP.cmake")
  include(UseHIP)
elseif(EXISTS "${HIP_PATH}/cmake/UseHIP.cmake")
  include(UseHIP)
endif()
# ROCm Thrust may include <cuda/__cccl_config>. Some targets (e.g. pure C++
# `paddle/pir`) compile without HIP-specific include order, so we must ensure
# vendored libcudacxx is visible *before* `${ROCM_PATH}/include/thrust`.
set(PADDLE_ROCM_LIBCUDACXX_INCLUDE_DIR
    "${PADDLE_SOURCE_DIR}/third_party/cccl/libcudacxx/include")
# Vendored libcudacxx first, then overlay patches (HIP `NV_IF_TARGET` fixes live in
# `patches/cccl/...`). CMake prepends each `BEFORE` at the front, so add the
# submodule path first, then the overlay — otherwise `third_party/cccl` wins
# and the overlay never takes effect (see compile_commands `-I` order).
if(EXISTS "${PADDLE_ROCM_LIBCUDACXX_INCLUDE_DIR}/cuda/__cccl_config")
  include_directories(BEFORE "${PADDLE_ROCM_LIBCUDACXX_INCLUDE_DIR}")
else()
  message(
    WARNING
      "Missing ${PADDLE_ROCM_LIBCUDACXX_INCLUDE_DIR}/cuda/__cccl_config. "
      "ROCm Thrust may fail to compile; initialize `third_party/cccl` (see cmake/external/cccl.cmake)."
  )
endif()
if(EXISTS "${PADDLE_SOURCE_DIR}/patches/cccl/libcudacxx/include/nv/target")
  include_directories(BEFORE
                      "${PADDLE_SOURCE_DIR}/patches/cccl/libcudacxx/include")
endif()
if(EXISTS "${PADDLE_SOURCE_DIR}/patches/thrust/thrust/shuffle.h")
  # Ensure our patched Thrust headers can override ROCm's Thrust when present.
  include_directories(BEFORE "${PADDLE_SOURCE_DIR}/patches/thrust")
endif()
if(EXISTS "${PADDLE_SOURCE_DIR}/patches/hip/fix_nv_if_target.h")
  # Allow patched Thrust shims to `#include "hip/fix_nv_if_target.h"`.
  include_directories(BEFORE "${PADDLE_SOURCE_DIR}/patches")
endif()
# Shadow select HIP headers so AMD platform macros are set before ROCm's
# `#error` paths. ROCm 7 HIPCC rules may not apply `HIP_CXX_FLAGS` / `-include`.
if(EXISTS "${PADDLE_SOURCE_DIR}/patches/rocm_shim/hip/hip_runtime.h")
  include_directories(BEFORE "${PADDLE_SOURCE_DIR}/patches/rocm_shim")
endif()
include_directories(${ROCM_PATH}/include)
message(STATUS "HIP version: ${HIP_VERSION}")
message(STATUS "HIP_CLANG_PATH: ${HIP_CLANG_PATH}")

macro(find_hip_version hip_header_file)
  file(READ ${hip_header_file} HIP_VERSION_FILE_CONTENTS)

  string(REGEX MATCH "define HIP_VERSION_MAJOR +([0-9]+)" HIP_MAJOR_VERSION
               "${HIP_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define HIP_VERSION_MAJOR +([0-9]+)" "\\1"
                       HIP_MAJOR_VERSION "${HIP_MAJOR_VERSION}")
  string(REGEX MATCH "define HIP_VERSION_MINOR +([0-9]+)" HIP_MINOR_VERSION
               "${HIP_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define HIP_VERSION_MINOR +([0-9]+)" "\\1"
                       HIP_MINOR_VERSION "${HIP_MINOR_VERSION}")
  string(REGEX MATCH "define HIP_VERSION_PATCH +([0-9]+)" HIP_PATCH_VERSION
               "${HIP_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define HIP_VERSION_PATCH +([0-9]+)" "\\1"
                       HIP_PATCH_VERSION "${HIP_PATCH_VERSION}")

  if(NOT HIP_MAJOR_VERSION)
    set(HIP_VERSION "???")
    message(
      WARNING "Cannot find HIP version in ${HIP_PATH}/include/hip/hip_version.h"
    )
  else()
    math(
      EXPR
      HIP_VERSION
      "${HIP_MAJOR_VERSION} * 10000000 + ${HIP_MINOR_VERSION} * 100000   + ${HIP_PATCH_VERSION}"
    )
    message(
      STATUS
        "Current HIP header is ${HIP_PATH}/include/hip/hip_version.h "
        "Current HIP version is v${HIP_MAJOR_VERSION}.${HIP_MINOR_VERSION}.${HIP_PATCH_VERSION}. "
    )
  endif()
endmacro()
set(HIP_VERSION_HEADER "")
if(EXISTS "${ROCM_PATH}/include/hip/hip_version.h")
  set(HIP_VERSION_HEADER "${ROCM_PATH}/include/hip/hip_version.h")
elseif(EXISTS "${ROCM_PATH}/hip/include/hip/hip_version.h")
  set(HIP_VERSION_HEADER "${ROCM_PATH}/hip/include/hip/hip_version.h")
elseif(EXISTS "${HIP_PATH}/include/hip/hip_version.h")
  set(HIP_VERSION_HEADER "${HIP_PATH}/include/hip/hip_version.h")
endif()
if(HIP_VERSION_HEADER)
  find_hip_version(${HIP_VERSION_HEADER})
else()
  message(
    WARNING
      "Cannot find hip_version.h under ${ROCM_PATH}/include/hip, ${ROCM_PATH}/hip/include/hip, or ${HIP_PATH}/include/hip"
  )
endif()

macro(find_package_and_include PACKAGE_NAME)
  find_package("${PACKAGE_NAME}" REQUIRED)
  include_directories("${ROCM_PATH}/${PACKAGE_NAME}/include")
  message(STATUS "${PACKAGE_NAME} version: ${${PACKAGE_NAME}_VERSION}")
endmacro()

find_package_and_include(miopen)
find_package_and_include(rocblas)
find_package_and_include(hipblaslt)
find_package_and_include(hiprand)
find_package_and_include(rocrand)
find_package_and_include(rccl)
find_package_and_include(rocthrust)
find_package_and_include(hipcub)
find_package_and_include(rocprim)
find_package_and_include(hipsparse)
find_package_and_include(rocsparse)
find_package_and_include(rocfft)
find_package_and_include(rocsolver)

if(CCACHE_PATH)
  set(HIP_HIPCC_EXECUTABLE ${CCACHE_PATH} ${HIP_HIPCC_EXECUTABLE})
endif()

# set CXX flags for HIP
set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__ -DROCM_NO_WRAPPER_HEADER_WARNING"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__ -DROCM_NO_WRAPPER_HEADER_WARNING"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP")
set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_HIP)

# define HIP_CXX_FLAGS
list(APPEND HIP_CXX_FLAGS -fPIC)
list(APPEND HIP_CXX_FLAGS -D__HIP_PLATFORM_HCC__=1)
list(APPEND HIP_CXX_FLAGS -D__HIP_PLATFORM_AMD__=1)
# Some ROCm headers (e.g., Thrust/NV_IF_TARGET) key off __HIPCC__ even when
# we compile HIP sources via clang++ -x hip instead of the hipcc wrapper.
list(APPEND HIP_CXX_FLAGS -D__HIPCC__=1)

# HIP public headers need a valid __HIP_PLATFORM_* before any `#include <hip/...>`.
# Some HIP-Clang invocations omit hipcc's default -D flags; force AMD ROCm first.
if(EXISTS "${PADDLE_SOURCE_DIR}/patches/hip/hip_amd_platform_prefix.h")
  list(APPEND HIP_CXX_FLAGS
              "-include${PADDLE_SOURCE_DIR}/patches/hip/hip_amd_platform_prefix.h")
endif()
# Note(qili93): HIP has compile conflicts of float16.h as platform::float16 overload std::is_floating_point and std::is_integer
list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_CONVERSIONS__=1)
list(APPEND HIP_CXX_FLAGS -DROCM_NO_WRAPPER_HEADER_WARNING)
list(APPEND HIP_CXX_FLAGS -Wno-macro-redefined)
list(APPEND HIP_CXX_FLAGS -Wno-inconsistent-missing-override)
list(APPEND HIP_CXX_FLAGS -Wno-exceptions)
list(APPEND HIP_CXX_FLAGS -Wno-shift-count-negative)
list(APPEND HIP_CXX_FLAGS -Wno-shift-count-overflow)
list(APPEND HIP_CXX_FLAGS -Wno-unused-command-line-argument)
list(APPEND HIP_CXX_FLAGS -Wno-duplicate-decl-specifier)
list(APPEND HIP_CXX_FLAGS -Wno-implicit-int-float-conversion)
list(APPEND HIP_CXX_FLAGS -Wno-pass-failed)
list(APPEND HIP_CXX_FLAGS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
# Avoid Thrust exceptions/throw paths in HIP builds (device code cannot throw).
list(APPEND HIP_CXX_FLAGS -DTHRUST_DISABLE_EXCEPTIONS=1)
list(APPEND HIP_CXX_FLAGS -DTHRUST_NO_EXCEPTIONS=1)
list(APPEND HIP_CXX_FLAGS -Wno-unused-result)
list(APPEND HIP_CXX_FLAGS -Wno-deprecated-declarations)
list(APPEND HIP_CXX_FLAGS -Wno-format)
list(APPEND HIP_CXX_FLAGS -Wno-dangling-gsl)
list(APPEND HIP_CXX_FLAGS -Wno-unused-value)
list(APPEND HIP_CXX_FLAGS -Wno-braced-scalar-init)
list(APPEND HIP_CXX_FLAGS -Wno-return-type)
list(APPEND HIP_CXX_FLAGS -Wno-pragma-once-outside-header)
list(APPEND HIP_CXX_FLAGS -Wno-deprecated-builtins)
list(APPEND HIP_CXX_FLAGS -Wno-switch)
list(APPEND HIP_CXX_FLAGS -Wno-literal-conversion)
list(APPEND HIP_CXX_FLAGS -Wno-constant-conversion)
list(APPEND HIP_CXX_FLAGS -Wno-defaulted-function-deleted)
list(APPEND HIP_CXX_FLAGS -Wno-sign-compare)
list(APPEND HIP_CXX_FLAGS -Wno-bitwise-instead-of-logical)
list(APPEND HIP_CXX_FLAGS -Wno-unknown-warning-option)
list(APPEND HIP_CXX_FLAGS -Wno-unused-lambda-capture)
list(APPEND HIP_CXX_FLAGS -Wno-unused-variable)
list(APPEND HIP_CXX_FLAGS -Wno-unused-but-set-variable)
list(APPEND HIP_CXX_FLAGS -Wno-reorder-ctor)
list(APPEND HIP_CXX_FLAGS -Wno-deprecated-copy-with-user-provided-copy)
list(APPEND HIP_CXX_FLAGS -Wno-unused-local-typedef)
list(APPEND HIP_CXX_FLAGS -Wno-missing-braces)
list(APPEND HIP_CXX_FLAGS -Wno-sometimes-uninitialized)
list(APPEND HIP_CXX_FLAGS -Wno-deprecated-copy)
list(APPEND HIP_CXX_FLAGS -Wno-pessimizing-move)
list(APPEND HIP_CXX_FLAGS -std=c++17)
list(APPEND HIP_CXX_FLAGS --gpu-max-threads-per-block=1024)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  list(APPEND HIP_CXX_FLAGS -g2)
  list(APPEND HIP_CXX_FLAGS -O0)
  list(APPEND HIP_HIPCC_FLAGS -fdebug-info-for-profiling)
endif()

set(HIP_HCC_FLAGS ${HIP_CXX_FLAGS})
set(HIP_CLANG_FLAGS ${HIP_CXX_FLAGS})
# Ask hcc to generate device code during compilation so we can use
# host linker to link.
list(APPEND HIP_HCC_FLAGS -fno-gpu-rdc)
list(APPEND HIP_HCC_FLAGS --offload-arch=gfx906) # Z100 (ZIFANG)
list(APPEND HIP_CLANG_FLAGS -fno-gpu-rdc)
list(APPEND HIP_CLANG_FLAGS --offload-arch=gfx906) # Z100 (ZIFANG)

if(HIP_COMPILER STREQUAL clang)
  set(hip_library_name amdhip64)
else()
  set(hip_library_name hip_hcc)
endif()
message(STATUS "HIP library name: ${hip_library_name}")

# set HIP link libs
find_library(ROCM_HIPRTC_LIB ${hip_library_name} HINTS ${ROCM_PATH}/lib
             ${ROCM_PATH}/lib64 ${HIP_PATH}/lib)
message(STATUS "ROCM_HIPRTC_LIB: ${ROCM_HIPRTC_LIB}")

include(thrust)

if(WITH_Z100)
  add_definitions(-DCINN_WITH_Z100)
endif()

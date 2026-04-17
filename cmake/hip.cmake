if(NOT WITH_ROCM)
  return()
endif()

if(NOT DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH
      "/opt/rocm"
      CACHE PATH "Path to which ROCm has been installed")
else()
  set(ROCM_PATH
      $ENV{ROCM_PATH}
      CACHE PATH "Path to which ROCm has been installed")
endif()

# ROCm 7.0+: HIP is now directly under ROCM_PATH, not in a separate hip subdirectory
# Check if we're using newer ROCm layout (7.0+) or older layout
if(EXISTS "${ROCM_PATH}/lib/cmake/hip/FindHIP.cmake")
  # ROCm 7.0+ layout
  set(HIP_PATH
      ${ROCM_PATH}
      CACHE PATH "Path to which HIP has been installed")
  set(CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
elseif(EXISTS "${ROCM_PATH}/hip/cmake")
  # Legacy ROCm layout (< 7.0)
  set(HIP_PATH
      ${ROCM_PATH}/hip
      CACHE PATH "Path to which HIP has been installed")
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
else()
  # Fallback: assume ROCm 7.0+ layout
  set(HIP_PATH
      ${ROCM_PATH}
      CACHE PATH "Path to which HIP has been installed")
  set(CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
endif()

set(HIP_CLANG_PATH
    ${ROCM_PATH}/llvm/bin
    CACHE PATH "Path to which clang has been installed")
set(CMAKE_PREFIX_PATH "${ROCM_PATH}" ${CMAKE_PREFIX_PATH})

find_package(HIP REQUIRED)
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
# ROCm 7.0+: hip_version.h is directly under ROCM_PATH/include
if(EXISTS "${ROCM_PATH}/include/hip/hip_version.h")
  find_hip_version(${ROCM_PATH}/include/hip/hip_version.h)
elseif(EXISTS "${HIP_PATH}/include/hip/hip_version.h")
  find_hip_version(${HIP_PATH}/include/hip/hip_version.h)
else()
  message(WARNING "Cannot find hip_version.h")
endif()

macro(find_package_and_include PACKAGE_NAME)
  find_package("${PACKAGE_NAME}" REQUIRED)
  # ROCm 7.0+ uses /opt/rocm/include/<package>/ instead of /opt/rocm/<package>/include/
  if(EXISTS "${ROCM_PATH}/include/${PACKAGE_NAME}")
    include_directories("${ROCM_PATH}/include/${PACKAGE_NAME}")
  elseif(EXISTS "${ROCM_PATH}/${PACKAGE_NAME}/include")
    include_directories("${ROCM_PATH}/${PACKAGE_NAME}/include")
  endif()
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
    "${CMAKE_C_FLAGS} -D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__ -D__HIP__=1 -DROCM_NO_WRAPPER_HEADER_WARNING"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__ -D__HIP__=1 -DROCM_NO_WRAPPER_HEADER_WARNING"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP")
set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_HIP)

# define HIP_CXX_FLAGS
list(APPEND HIP_CXX_FLAGS -fPIC)
list(APPEND HIP_CXX_FLAGS -D__HIP_PLATFORM_HCC__=1)
list(APPEND HIP_CXX_FLAGS -D__HIP_PLATFORM_AMD__=1)
list(APPEND HIP_CXX_FLAGS -D__HIP__=1)
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
list(APPEND HIP_HCC_FLAGS --offload-arch=gfx942) # MI300
list(APPEND HIP_HCC_FLAGS --offload-arch=gfx950) # MI350X
list(APPEND HIP_CLANG_FLAGS -fno-gpu-rdc)
list(APPEND HIP_CLANG_FLAGS --offload-arch=gfx942) # MI300
list(APPEND HIP_CLANG_FLAGS --offload-arch=gfx950) # MI350X

if(HIP_COMPILER STREQUAL clang)
  set(hip_library_name amdhip64)
else()
  set(hip_library_name hip_hcc)
endif()
message(STATUS "HIP library name: ${hip_library_name}")

# set HIP link libs
find_library(ROCM_HIPRTC_LIB ${hip_library_name} HINTS ${HIP_PATH}/lib)
message(STATUS "ROCM_HIPRTC_LIB: ${ROCM_HIPRTC_LIB}")

include(thrust)

if(WITH_Z100)
  add_definitions(-DCINN_WITH_Z100)
endif()

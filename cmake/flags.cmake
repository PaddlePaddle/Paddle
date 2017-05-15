# Setting Paddle Compile Flags
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)
include(CheckCXXSymbolExists)
include(CheckTypeSize)

function(CheckCompilerCXX11Flag)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 4.8)
            message(FATAL_ERROR "Unsupported GCC version. GCC >= 4.8 required.")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # cmake >= 3.0 compiler id "AppleClang" on Mac OS X, otherwise "Clang"
        # Apple Clang is a different compiler than upstream Clang which havs different version numbers.
        # https://gist.github.com/yamaya/2924292
        if(APPLE)  # cmake < 3.0 compiler id "Clang" on Mac OS X
            if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 5.1)
                message(FATAL_ERROR "Unsupported AppleClang version. AppleClang >= 5.1 required.")
            endif()
        else()
            if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.3)
                message(FATAL_ERROR "Unsupported Clang version. Clang >= 3.3 required.")
            endif()
        endif()   
    endif()
endfunction()

CheckCompilerCXX11Flag()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# safe_set_flag
#
# Set a compile flag only if compiler is support
# is_c: is C flag or C++ flag, bool type.
# src_list: The list name which the flag name will be append to.
# flag_name: the flag name for compiler, such as '-Werror' '-Wall' etc
# rest arguments: not used.
function(safe_set_flag is_c src_list flag_name)
    string(REPLACE "-" "_" safe_name ${flag_name})
    string(REPLACE "=" "_" safe_name ${safe_name})
    if(is_c)
        CHECK_C_COMPILER_FLAG(${flag_name} C_COMPILER_SUPPORT_FLAG_${safe_name})
        set(safe_name C_COMPILER_SUPPORT_FLAG_${safe_name})
    else()
        CHECK_CXX_COMPILER_FLAG(${flag_name} CXX_COMPILER_SUPPORT_FLAG_${safe_name})
        set(safe_name CXX_COMPILER_SUPPORT_FLAG_${safe_name})
    endif()
    if(${safe_name})
        set(${src_list} "${${src_list}} ${flag_name}" PARENT_SCOPE)
    endif()
endfunction()

# helper macro to set cflag
macro(safe_set_cflag src_list flag_name)
    safe_set_flag(ON ${src_list} ${flag_name})
endmacro()

# helper macro to set cxxflag
macro(safe_set_cxxflag src_list flag_name)
    safe_set_flag(OFF ${src_list} ${flag_name})
endmacro()

# helper macro to set nvcc flag
macro(safe_set_nvflag flag_name)
    string(REPLACE "-" "_" safe_name ${flag_name})
    string(REPLACE "=" "_" safe_name ${safe_name})
    CHECK_C_COMPILER_FLAG(${flag_name} C_COMPILER_SUPPORT_FLAG_${safe_name})
    set(safe_name C_COMPILER_SUPPORT_FLAG_${safe_name})
    if(${safe_name})
        LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler ${flag_name})
    endif()
endmacro()


CHECK_CXX_SYMBOL_EXISTS(UINT64_MAX "stdint.h" UINT64_MAX_EXISTS)
if(NOT UINT64_MAX_EXISTS)
  set(CMAKE_REQUIRED_DEFINITIONS -D__STDC_LIMIT_MACROS)
  CHECK_CXX_SYMBOL_EXISTS(UINT64_MAX "stdint.h" UINT64_MAX_EXISTS_HERE)
  if(UINT64_MAX_EXISTS_HERE)
    set(CMAKE_REQUIRED_DEFINITIONS)
    add_definitions(-D__STDC_LIMIT_MACROS)
  else()
    message(FATAL_ERROR "Cannot find symbol UINT64_MAX")
  endif()
endif()

SET(CMAKE_EXTRA_INCLUDE_FILES "pthread.h")
CHECK_TYPE_SIZE(pthread_spinlock_t SPINLOCK_FOUND)
CHECK_TYPE_SIZE(pthread_barrier_t BARRIER_FOUND)
if(SPINLOCK_FOUND)
  add_definitions(-DPADDLE_USE_PTHREAD_SPINLOCK)
endif(SPINLOCK_FOUND)
if(BARRIER_FOUND)
  add_definitions(-DPADDLE_USE_PTHREAD_BARRIER)
endif(BARRIER_FOUND)
SET(CMAKE_EXTRA_INCLUDE_FILES "")

# Common flags. the compiler flag used for C/C++ sources whenever release or debug
# Do not care if this flag is support for gcc.
set(COMMON_FLAGS
    -fPIC
    -fno-omit-frame-pointer
    -Wall
    -Wextra
    -Werror
    -Wnon-virtual-dtor
    -Wdelete-non-virtual-dtor
    -Wno-unused-parameter
    -Wno-unused-function
    -Wno-error=literal-suffix
    -Wno-error=sign-compare
    -Wno-error=unused-local-typedefs)

set(GPU_COMMON_FLAGS
    -fPIC
    -fno-omit-frame-pointer
    -Wnon-virtual-dtor
    -Wdelete-non-virtual-dtor
    -Wno-unused-parameter
    -Wno-unused-function
    -Wno-error=sign-compare
    -Wno-error=literal-suffix
    -Wno-error=unused-local-typedefs
    -Wno-error=unused-function  # Warnings in Numpy Header.
)

if (APPLE)
    # On Mac OS X build fat binaries with x86_64 architectures by default.
    set (CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Build architectures for OSX" FORCE)
else()
    set(GPU_COMMON_FLAGS
        -Wall
        -Wextra
        -Werror
        ${GPU_COMMON_FLAGS})
endif()


foreach(flag ${COMMON_FLAGS})
    safe_set_cflag(CMAKE_C_FLAGS ${flag})
    safe_set_cxxflag(CMAKE_CXX_FLAGS ${flag})
endforeach()

foreach(flag ${GPU_COMMON_FLAGS})
    safe_set_nvflag(${flag})
endforeach()


set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Release/Debug flags set by cmake. Such as -O3 -g -DNDEBUG etc.
# So, don't set these flags here.
LIST(APPEND CUDA_NVCC_FLAGS -std=c++11)
LIST(APPEND CUDA_NVCC_FLAGS --use_fast_math)

if(CMAKE_BUILD_TYPE  STREQUAL "Debug")
    LIST(APPEND CUDA_NVCC_FLAGS  ${CMAKE_CXX_FLAGS_DEBUG})
elseif(CMAKE_BUILD_TYPE  STREQUAL "Release")
    LIST(APPEND CUDA_NVCC_FLAGS  ${CMAKE_CXX_FLAGS_RELEASE})
elseif(CMAKE_BUILD_TYPE  STREQUAL "RelWithDebInfo")
    LIST(APPEND CUDA_NVCC_FLAGS  ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
elseif(CMAKE_BUILD_TYPE  STREQUAL "MinSizeRel")
    LIST(APPEND CUDA_NVCC_FLAGS  ${CMAKE_CXX_FLAGS_MINSIZEREL})
endif()

function(specify_cuda_arch cuda_version cuda_arch)
    if(${cuda_version} VERSION_GREATER "8.0")
        foreach(capability 61 62)
          if(${cuda_arch} STREQUAL ${capability})
            list(APPEND __arch_flags " -gencode arch=compute_${cuda_arch},code=sm_${cuda_arch}")
          endif()
        endforeach()
    elseif(${cuda_version} VERSION_GREATER "7.0" and ${cuda_arch} STREQUAL "53")
        list(APPEND __arch_flags " -gencode arch=compute_${cuda_arch},code=sm_${cuda_arch}")
    endif()
endfunction()

# Common gpu architectures: Kepler, Maxwell
foreach(capability 30 35 50)
      list(APPEND __arch_flags " -gencode arch=compute_${capability},code=sm_${capability}")
endforeach()

if (CUDA_VERSION VERSION_GREATER "7.0" OR CUDA_VERSION VERSION_EQUAL "7.0")
      list(APPEND __arch_flags " -gencode arch=compute_52,code=sm_52")
endif()

# Modern gpu architectures: Pascal
if (CUDA_VERSION VERSION_GREATER "8.0" OR CUDA_VERSION VERSION_EQUAL "8.0")
      list(APPEND __arch_flags " -gencode arch=compute_60,code=sm_60")
endif()

# Custom gpu architecture
set(CUDA_ARCH)

if(CUDA_ARCH)
  specify_cuda_arch(${CUDA_VERSION} ${CUDA_ARCH})
endif()

set(CUDA_NVCC_FLAGS ${__arch_flags} ${CUDA_NVCC_FLAGS})


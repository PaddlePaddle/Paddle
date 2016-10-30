# Setting Paddle Compile Flags
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)
include(CheckCXXSymbolExists)
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
        if(is_c)
          set(CUDA_NVCC_FLAGS
              --compiler-options;${flag_name}
              ${CUDA_NVCC_FLAGS}
              PARENT_SCOPE)
        endif()
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
    -Wno-error=literal-suffix
    -Wno-error=unused-local-typedefs
    -Wno-error=unused-function  # Warnings in Numpy Header.
)

foreach(flag ${COMMON_FLAGS})
    safe_set_cflag(CMAKE_C_FLAGS ${flag})
    safe_set_cxxflag(CMAKE_CXX_FLAGS ${flag})
endforeach()

# On Mac OS X build fat binaries with x86_64 architectures by default.
if (APPLE)
    set (CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Build architectures for OSX" FORCE)
endif ()

# Release/Debug flags set by cmake. Such as -O3 -g -DNDEBUG etc.
# So, don't set these flags here.

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

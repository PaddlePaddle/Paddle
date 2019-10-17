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

    if(${flag_name} MATCHES "fsanitize")
        set(CMAKE_REQUIRED_FLAGS_RETAINED ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS ${flag_name})
    endif()

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

    if(${flag_name} MATCHES "fsanitize")
        set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_RETAINED})
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

macro(safe_set_static_flag) # set c_flags and cxx_flags to static or shared
    if (BUILD_SHARED_LIBS) 
        return() # if build shared libs, the flags keep same with '/MD'
    endif(BUILD_SHARED_LIBS)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
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

# Only one sanitizer is allowed in compile time
string(TOLOWER "${SANITIZER_TYPE}" sanitizer_type)
if(sanitizer_type STREQUAL "address")
    set(fsanitize "-fsanitize=address")
elseif(sanitizer_type STREQUAL "leak")
    set(fsanitize "-fsanitize=leak")
elseif(sanitizer_type STREQUAL "memory")
    set(fsanitize "-fsanitize=memory")
elseif(sanitizer_type STREQUAL "thread")
    set(fsanitize "-fsanitize=thread")
elseif(sanitizer_type STREQUAL "undefined")
    set(fsanitize "-fsanitize=undefined")
endif()

# Common flags. the compiler flag used for C/C++ sources whenever release or debug
# Do not care if this flag is support for gcc.

# https://github.com/PaddlePaddle/Paddle/issues/12773
if (NOT WIN32)
set(COMMON_FLAGS
    -fPIC
    -fno-omit-frame-pointer
    -Werror
    -Wall
    -Wextra
    -Wnon-virtual-dtor
    -Wdelete-non-virtual-dtor
    -Wno-unused-parameter
    -Wno-unused-function
    -Wno-error=literal-suffix
    -Wno-error=sign-compare
    -Wno-error=unused-local-typedefs
    -Wno-error=parentheses-equality # Warnings in pybind11
    -Wno-error=ignored-attributes  # Warnings in Eigen, gcc 6.3
    -Wno-error=terminate  # Warning in PADDLE_ENFORCE
    -Wno-error=int-in-bool-context # Warning in Eigen gcc 7.2
    -Wimplicit-fallthrough=0 # Warning in tinyformat.h
    ${fsanitize}
)

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
    -Wno-error=array-bounds # Warnings in Eigen::array
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
endif(NOT WIN32)

if (APPLE)
    # On Mac OS X build fat binaries with x86_64 architectures by default.
    set (CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Build architectures for OSX" FORCE)
    # On Mac OS X register class specifier is deprecated and will cause warning error on latest clang 10.0
    set (COMMON_FLAGS -Wno-deprecated-register)
endif(APPLE)

if(LINUX)
    set(GPU_COMMON_FLAGS
        -Wall
        -Wextra
        -Werror
        ${GPU_COMMON_FLAGS})
endif(LINUX)

if(UNIX AND NOT APPLE)
  # except apple from nix*Os family
  set(LINUX TRUE)
endif(UNIX AND NOT APPLE)

foreach(flag ${COMMON_FLAGS})
    safe_set_cflag(CMAKE_C_FLAGS ${flag})
    safe_set_cxxflag(CMAKE_CXX_FLAGS ${flag})
endforeach()

foreach(flag ${GPU_COMMON_FLAGS})
    safe_set_nvflag(${flag})
endforeach()

if(WIN32 AND MSVC_STATIC_CRT)
# windows build turn off warnings.
safe_set_static_flag()
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
        string(REGEX REPLACE "(^| )/W[0-9]( |$)" " " ${flag_var} "${${flag_var}}")
        set(flag_var "${flag_var} /w")
    endforeach(flag_var)
endif()

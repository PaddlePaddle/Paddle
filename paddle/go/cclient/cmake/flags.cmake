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

set(CUDA_NVCC_FLAGS ${__arch_flags} ${CUDA_NVCC_FLAGS})
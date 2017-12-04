if(NOT WITH_GPU)
    return()
endif()


include(FindHIP)

include_directories("/opt/rocm/hip/include")
include_directories("/usr/include/x86_64-linux-gnu")


# Release/Debug flags set by cmake. Such as -O3 -g -DNDEBUG etc.
# So, don't set these flags here.
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
# Set :expt-relaxed-constexpr to suppress Eigen warnings
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "-D__HIP_PLATFORM_HCC__")
#list(APPEND CUDA_NVCC_FLAGS ${nvcc_flags})
list(APPEND CUDA_NVCC_FLAGS "-DPADDLE_USE_DSO -DPADDLE_DISABLE_TIMER -DPADDLE_DISABLE_PROFILER -DPADDLE_WITHOUT_GOLANG -DPADDLE_WITH_CUDA -DPADDLE_DISABLE_RDMA -DPADDLE_USE_PTHREAD_SPINLOCK -DPADDLE_USE_PTHREAD_BARRIER -DPADDLE_VERSION=0.10.0")

if(CMAKE_BUILD_TYPE  STREQUAL "Debug")
    list(APPEND CUDA_NVCC_FLAGS  ${CMAKE_CXX_FLAGS_DEBUG})
elseif(CMAKE_BUILD_TYPE  STREQUAL "Release")
    list(APPEND CUDA_NVCC_FLAGS  ${CMAKE_CXX_FLAGS_RELEASE})
elseif(CMAKE_BUILD_TYPE  STREQUAL "RelWithDebInfo")
    list(APPEND CUDA_NVCC_FLAGS  ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
elseif(CMAKE_BUILD_TYPE  STREQUAL "MinSizeRel")
    list(APPEND CUDA_NVCC_FLAGS  ${CMAKE_CXX_FLAGS_MINSIZEREL})
endif()

mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)
mark_as_advanced(CUDA_SDK_ROOT_DIR CUDA_SEPARABLE_COMPILATION)

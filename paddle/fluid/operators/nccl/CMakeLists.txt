if(NOT (WITH_NCCL OR WITH_RCCL))
  return()
endif()

if(WITH_GPU AND NOT WIN32)
  nv_library(
    nccl_common
    SRCS nccl_gpu_common.cc
    DEPS device_context operator)
endif()

if(WITH_ROCM AND NOT WIN32)
  hip_library(
    nccl_common
    SRCS nccl_gpu_common.cc
    DEPS device_context operator)
endif()

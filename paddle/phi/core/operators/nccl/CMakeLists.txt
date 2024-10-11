if(NOT (WITH_NCCL OR WITH_RCCL))
  return()
endif()

if(WITH_GPU AND NOT WIN32)
  collect_srcs(core_srcs SRCS nccl_gpu_common.cc)
endif()

if(WITH_ROCM AND NOT WIN32)
  collect_srcs(core_srcs SRCS nccl_gpu_common.cc)
endif()

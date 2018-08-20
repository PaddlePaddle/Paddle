if(WITH_GPU AND NOT WIN32)
  nv_library(nccl_common SRCS nccl_gpu_common.cc DEPS device_context operator )
endif()

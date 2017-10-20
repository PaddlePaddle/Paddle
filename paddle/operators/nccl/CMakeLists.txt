if(WITH_GPU)
  nv_library(nccl_common SRCS nccl_gpu_common.cc DEPS device_context operator)
  nv_test(nccl_gpu_common_test SRCS nccl_gpu_common_test.cc DEPS nccl_common)
endif()

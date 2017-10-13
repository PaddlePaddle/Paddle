if(WITH_GPU)
  nv_library(nccl_common SRCS nccl_gpu_common DEPS device_context operator)
  nv_library(nccl_op SRCS nccl_ops.cc DEPS nccl_common)
else()
  cc_library(nccl_common SRCS nccl_gpu_common DEPS device_context operator)
endif()

cc_test(nccl_gpu_common_test SRCS nccl_gpu_common_test.cc DEPS nccl_common)

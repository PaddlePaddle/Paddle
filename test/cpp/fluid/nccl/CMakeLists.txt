if(WITH_GPU AND NOT WIN32)
  nv_test(
    nccl_op_test
    SRCS nccl_op_test.cu.cc
    DEPS nccl_op gpu_info device_context)
endif()

if(WITH_ROCM AND NOT WIN32)
  hip_test(
    nccl_op_test
    SRCS nccl_op_test.cu.cc
    DEPS nccl_op gpu_info device_context)
endif()

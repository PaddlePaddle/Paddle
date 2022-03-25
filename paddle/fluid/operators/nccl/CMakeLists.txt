if (NOT (WITH_NCCL OR WITH_RCCL))
  return()
endif()

if(WITH_GPU AND NOT WIN32)
  nv_library(nccl_common SRCS nccl_gpu_common.cc DEPS device_context operator )
endif()

if(WITH_ROCM AND NOT WIN32)
  hip_library(nccl_common SRCS nccl_gpu_common.cc DEPS device_context operator )
endif()

if(WITH_GPU OR WITH_ROCM)
    op_library(nccl_op DEPS nccl_common)
    set(OPERATOR_DEPS ${OPERATOR_DEPS} nccl_common PARENT_SCOPE)
endif()

if(WITH_GPU AND NOT WIN32)
    nv_test(nccl_op_test SRCS nccl_op_test.cu.cc DEPS nccl_op gpu_info device_context)
endif()

if(WITH_ROCM AND NOT WIN32)
    hip_test(nccl_op_test SRCS nccl_op_test.cu.cc DEPS nccl_op gpu_info device_context)
endif()

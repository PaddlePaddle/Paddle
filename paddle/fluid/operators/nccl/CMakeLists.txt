if (NOT WITH_NCCL)
  return()
endif()

if(WITH_GPU AND NOT WIN32)
  nv_library(nccl_common SRCS nccl_gpu_common.cc DEPS device_context operator )
endif()

if(WITH_GPU)
    op_library(nccl_op DEPS nccl_common)
    file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(ncclAllReduce);\n")
    set(OPERATOR_DEPS ${OPERATOR_DEPS} nccl_common PARENT_SCOPE)
endif()

if(NOT WIN32)
    nv_test(nccl_op_test SRCS nccl_op_test.cu.cc DEPS nccl_op gpu_info device_context)
endif()

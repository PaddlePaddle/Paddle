if(WITH_ASCEND_CL)
  cc_library(npu_hccl SRCS hccl.cc DEPS dynamic_loader warpctc)
endif()

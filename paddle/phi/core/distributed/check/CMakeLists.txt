cc_library(
  comm_static_check
  SRCS static_check.cc
  DEPS place dense_tensor enforce)

if(WITH_NCCL OR WITH_RCCL)
  cc_library(
    nccl_dynamic_check
    SRCS nccl_dynamic_check.cc
    DEPS dense_tensor)
endif()

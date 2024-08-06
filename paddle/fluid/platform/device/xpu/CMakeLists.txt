if(NOT WITH_XPU)
  return()
endif()

set(XPU_CTX_DEPS
    xpulib
    ssl
    crypto
    rt
    z
    resolv
    dl)

cc_library(
  xpu_op_list
  SRCS xpu_op_list.cc
  DEPS glog xpulib device_context op_kernel_type phi common)
cc_library(
  xpu_resource_pool
  SRCS xpu_resource_pool.cc
  DEPS phi common)

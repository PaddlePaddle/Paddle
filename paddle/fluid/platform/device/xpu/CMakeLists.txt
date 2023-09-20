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
  xpu_info
  SRCS xpu_info.cc
  DEPS glog
       enforce
       xpulib
       device_context
       place
       phi
       dynload_xpti)
cc_library(
  xpu_op_list
  SRCS xpu_op_list.cc
  DEPS glog
       enforce
       xpulib
       device_context
       op_kernel_type
       phi
       dynload_xpti)
cc_library(
  xpu_resource_pool
  SRCS xpu_resource_pool.cc
  DEPS xpu_info)

if(NOT WITH_XPU)
  return()
endif()

core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  xpu_module.cc
  xpu_util.cc
  xpu_backend_api.cc
  xpu_intrinsics.cc
  xpu_intrinsics_reduce.cc
  xpu_intrinsics_float16.cc
  xpu_intrinsics_bfloat16.cc)

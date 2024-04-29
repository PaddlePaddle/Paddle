if(NOT WITH_CUDA)
  return()
endif()

core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  cuda_module.cc
  cuda_util.cc
  cuda_intrinsics.cc
  cuda_intrinsics_reduce.cc
  cuda_instrinsics_float16.cc
  cuda_instrinsics_bfloat16.cc)

cinn_nv_test(test_cuda_module SRCS cuda_module_test.cc DEPS cinncore)
cinn_nv_library(cuda_runtime SRCS cinn_cuda_runtime_source.cuh)

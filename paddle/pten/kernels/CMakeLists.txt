# kernel primitive api
add_subdirectory(primitive)
# pten hybird functors and functions called by kernels
add_subdirectory(hybird)

add_subdirectory(cpu)
if(WITH_GPU OR WITH_ROCM)
  # NOTE(chenweihang): if hip can split from cuda impl, we should add hip dir
  add_subdirectory(cuda)
endif()
if(WITH_MKLDNN)
  # mkldnn will be deprecated and use the new name dnnl
  add_subdirectory(dnnl)
endif()
if(WITH_ASCEND_CL)
  add_subdirectory(npu)
endif()
if(WITH_XPU)
  add_subdirectory(xpu)
endif()

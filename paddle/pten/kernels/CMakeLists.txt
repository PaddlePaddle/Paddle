# kernel primitive api
add_subdirectory(primitive)
# pten hybird functors and functions called by kernels
add_subdirectory(hybird)

add_subdirectory(cpu)
if(WITH_GPU OR WITH_ROCM)
  add_subdirectory(gpu)
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

set(FLATTEN_DEPS dense_tensor kernel_context kernel_factory utils_cpu unary)
if(WITH_GPU OR WITH_ROCM)
  set(FLATTEN_DEPS ${FLATTEN_DEPS} utils_gpu)
elseif(WITH_XPU)
  set(FLATTEN_DEPS ${FLATTEN_DEPS} utils_xpu)
endif()

if(WITH_GPU)
  nv_library(flatten SRCS flatten_kernel.cc DEPS ${FLATTEN_DEPS})
elseif(WITH_ROCM)
  hip_library(flatten SRCS flatten_kernel.cc DEPS ${FLATTEN_DEPS})
else()
  cc_library(flatten SRCS flatten_kernel.cc DEPS ${FLATTEN_DEPS})
endif()

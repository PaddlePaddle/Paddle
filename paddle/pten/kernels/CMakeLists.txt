# kernel primitive api
add_subdirectory(primitive)
# pten hybird functors and functions called by kernels
add_subdirectory(hybird)

# pten kernels for different backends
# NOTE(chenweihang): We need to increase the compilation option of WITH_EIGEN,
# which will support splitting eigen at compile time on demand in the future
add_subdirectory(eigen)
# NOTE(chenweihang): We need to increase the compilation option of WITH_BLAS,
# which will support splitting eigen at compile time on demand in the future,
# and if necessary, blas can be split into openblas and cublas
add_subdirectory(blas)
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

if(WITH_GPU OR WITH_ROCM)
  set(FLATTEN_DEPS dense_tensor kernel_context kernel_factory utils_cuda unary)
elseif(WITH_XPU)
  set(FLATTEN_DEPS dense_tensor kernel_context kernel_factory utils_xpu unary)
else()
  set(FLATTEN_DEPS dense_tensor kernel_context kernel_factory utils_cpu unary)
endif()

if(WITH_GPU)
  nv_library(flatten SRCS flatten_kernel.cc DEPS ${FLATTEN_DEPS})
elseif(WITH_ROCM)
  hip_library(flatten SRCS flatten_kernel.cc DEPS ${FLATTEN_DEPS})
else()
  cc_library(flatten SRCS flatten_kernel.cc DEPS ${FLATTEN_DEPS})
endif()

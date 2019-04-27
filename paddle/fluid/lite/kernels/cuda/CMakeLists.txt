nv_library(mul_compute_cuda SRCS mul_compute.cc DEPS tensor_lite)
cc_library(io_copy_compute_cuda SRCS io_copy_compute.cc DEPS tensor_lite)

nv_library(kernels_cuda DEPS mul_compute_cuda io_copy_compute_cuda cuda_blas_lite)

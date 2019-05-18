if(NOT LITE_WITH_CUDA)
    return()
endif()

nv_library(target_wrapper_cuda SRCS target_wrapper.cc)
nv_library(cuda_blas_lite SRCS blas.cc)

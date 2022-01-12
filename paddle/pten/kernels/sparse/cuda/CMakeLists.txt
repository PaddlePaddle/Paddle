if(WITH_GPU)
    nv_library(sparse_csr_tensor_utils_cuda SRCS sparse_csr_tensor_utils.cu DEPS dense_tensor sparse_csr_tensor kernel_context kernel_factory device_context blas memory convert_utils)
    nv_library(sparse_coo_tensor_utils_cuda SRCS sparse_coo_tensor_utils.cu DEPS dense_tensor sparse_coo_tensor kernel_context kernel_factory device_context blas memory convert_utils)
endif()

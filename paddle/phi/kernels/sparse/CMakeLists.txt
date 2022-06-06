set(SPARSE_KERNEL_DEPS
    dense_tensor
    sparse_coo_tensor
    sparse_csr_tensor
    kernel_context
    kernel_factory
    arg_map_context
    convert_utils
    lod_utils
    math_function
    custom_kernel
    copy_kernel)

set(MANUAL_BUILD_KERNELS sparse_mm_kernel sparse_mm_grad_kernel)

register_kernels(EXCLUDES ${MANUAL_BUILD_KERNELS} DEPS ${SPARSE_KERNEL_DEPS}
                 SUB_DIR "sparse")

if(NOT WIN32 AND (CUDA_VERSION GREATER_EQUAL 11.2))
  kernel_library(sparse_mm_kernel DEPS ${SPARSE_KERNEL_DEPS})
  kernel_library(sparse_mm_grad_kernel DEPS ${SPARSE_KERNEL_DEPS})
endif()


set(SPARSE_KERNEL_DEPS dense_tensor sparse_coo_tensor sparse_csr_tensor kernel_context kernel_factory arg_map_context convert_utils lod_utils)
register_kernels(DEPS ${SPARSE_KERNEL_DEPS} SUB_DIR "sparse_kernel")

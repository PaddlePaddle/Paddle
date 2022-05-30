add_subdirectory(cpu)
if(WITH_GPU OR WITH_ROCM)
  add_subdirectory(gpu)
endif()

cc_library(unicode SRCS unicode.cc DEPS utf8proc)
set_property(GLOBAL PROPERTY STRING_KERNELS "")

set(STRING_KERNEL_DEPS dense_tensor string_tensor sparse_coo_tensor sparse_csr_tensor kernel_context kernel_factory arg_map_context convert_utils lod_utils custom_kernel)
set(STRING_KERNEL_DEPS ${STRING_KERNEL_DEPS} eigen_function blas math_function)
# remove this dep after removing fluid deps on tensor creation
set(STRING_KERNEL_DEPS ${STRING_KERNEL_DEPS} phi_api_utils)
set(STRING_KERNEL_DEPS ${STRING_KERNEL_DEPS} string_infermeta)
set(STRING_KERNEL_DEPS ${STRING_KERNEL_DEPS} unicode)

register_kernels(DEPS ${STRING_KERNEL_DEPS} SUB_DIR "strings")

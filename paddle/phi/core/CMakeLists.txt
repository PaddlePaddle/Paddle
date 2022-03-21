# compatible utils used for fluid op system
add_subdirectory(compat)

cc_library(errors SRCS errors.cc)
set(phi_enforce_deps errors flags)
if(WITH_GPU)
  set(phi_enforce_deps ${phi_enforce_deps} external_error_proto)
endif()
cc_library(phi_enforce INTERFACE SRCS enforce.cc DEPS ${phi_enforce_deps})

cc_library(kernel_factory SRCS kernel_factory.cc DEPS phi_enforce fluid_convert_utils)
cc_library(kernel_context SRCS kernel_context.cc DEPS phi_enforce phi_context)

cc_library(ddim SRCS ddim.cc DEPS phi_enforce)
cc_library(tensor_base SRCS tensor_base.cc allocator.cc DEPS phi_enforce)
cc_library(tensor_meta SRCS tensor_meta.cc DEPS phi_enforce)
cc_library(lod_utils SRCS lod_utils.cc DEPS phi_enforce)

cc_library(dense_tensor SRCS dense_tensor.cc dense_tensor_impl.cc DEPS fluid_convert_utils tensor_meta tensor_base)
cc_library(sparse_coo_tensor SRCS sparse_coo_tensor.cc DEPS tensor_meta tensor_base)
cc_library(sparse_csr_tensor SRCS sparse_csr_tensor.cc DEPS dense_tensor tensor_base)

cc_library(meta_tensor SRCS meta_tensor.cc DEPS tensor_base tensor_meta dense_tensor)
cc_library(infermeta_utils SRCS infermeta_utils.cc DEPS meta_tensor)
cc_library(selected_rows SRCS selected_rows_impl.cc DEPS dense_tensor phi_enforce ddim memcpy)
cc_library(phi_device_context SRCS device_context.cc DEPS dense_tensor selected_rows)

cc_library(custom_kernel SRCS custom_kernel.cc DEPS kernel_factory)

# Will remove once we implemented MKLDNN_Tensor
if(WITH_MKLDNN)
    add_dependencies(dense_tensor mkldnn)
    add_dependencies(tensor_base mkldnn)
endif()

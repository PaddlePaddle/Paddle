if(WITH_GPU)
  cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place gpu_info)
elseif(WITH_ROCM)
  cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place gpu_info)
else()
  cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place)
endif()

set(pten_enforce_deps errors flags)
cc_library(pten_enforce INTERFACE SRCS enforce.cc DEPS ${pten_enforce_deps})

cc_library(kernel_factory SRCS kernel_factory.cc DEPS pten_enforce convert_utils)
cc_library(kernel_context SRCS kernel_context.cc DEPS pten_enforce pten_context)
cc_library(arg_map_context SRCS arg_map_context.cc DEPS pten_enforce)

cc_library(tensor_base SRCS tensor_base.cc allocator.cc storage.cc DEPS pten_enforce)
cc_library(tensor_meta SRCS tensor_meta.cc DEPS pten_enforce mixed_vector)
cc_library(dense_tensor SRCS dense_tensor.cc DEPS convert_utils tensor_meta tensor_base)
cc_library(errors SRCS errors.cc)

# Will remove once we implemented MKLDNN_Tensor
if(WITH_MKLDNN)
    add_dependencies(dense_tensor mkldnn)
endif()

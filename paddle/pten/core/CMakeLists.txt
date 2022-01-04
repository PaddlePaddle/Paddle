if(WITH_GPU)
  cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place gpu_info)
elseif(WITH_ROCM)
  cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place gpu_info)
else()
  cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place)
endif()

cc_library(kernel_factory SRCS kernel_factory.cc DEPS enforce convert_utils)
cc_library(kernel_context SRCS kernel_context.cc DEPS enforce pten_context)
cc_library(tensor_base SRCS tensor_base.cc allocator.cc storage.cc DEPS enforce)

cc_library(tensor_meta SRCS tensor_meta.cc DEPS enforce mixed_vector)
cc_library(dense_tensor SRCS dense_tensor.cc DEPS convert_utils tensor_meta tensor_base)

# Will remove once we implemented MKLDNN_Tensor
if(WITH_MKLDNN)
    add_dependencies(dense_tensor mkldnn)
endif()

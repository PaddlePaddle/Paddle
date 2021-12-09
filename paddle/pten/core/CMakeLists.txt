if(WITH_GPU)
  cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place gpu_info)
elseif(WITH_ROCM)
  cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place gpu_info)
else()
  cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place)
endif()

cc_library(kernel_factory SRCS kernel_factory.cc DEPS enforce convert_utils)
cc_library(kernel_context SRCS kernel_context.cc DEPS enforce device_context)

cc_library(tensor_base SRCS tensor_base.cc allocator.cc storage.cc DEPS enforce)
cc_library(tensor_meta SRCS tensor_meta.cc DEPS enforce)
cc_library(dense_tensor SRCS dense_tensor.cc DEPS tensor_meta tensor_base)

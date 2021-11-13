IF(WITH_MKLDNN)
    set(MKLDNN_CTX_DEPS mkldnn)
ELSE()
    set(MKLDNN_CTX_DEPS)
ENDIF()

if(WITH_GPU)
  cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place gpu_info)
elseif(WITH_ROCM)
  cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place gpu_info)
else()
  cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place)
endif()

cc_library(device_context_base SRCS device_context.cc DEPS place)
# pten device context
add_subdirectory(backends)
cc_library(device_context_pool_base SRCS device_context_pool.cc DEPS device_context_base place)
set(pten_ctx device_context_base device_context_pool_base ${backends_context} PARENT_SCOPE)
if(WITH_GPU OR WITH_ROCM)
  set(pten_ctx ${pten_ctx} cuda_context PARENT_SCOPE)
endif()

cc_library(kernel_factory SRCS kernel_factory.cc DEPS enforce)
cc_library(kernel_context SRCS kernel_context.cc DEPS enforce device_context_base)

cc_library(tensor_base SRCS tensor_base.cc allocator.cc storage.cc DEPS enforce)
cc_library(dense_tensor SRCS dense_tensor.cc DEPS tensor_base)

if (WITH_GPU)
  cc_library(cuda_context SRCS cuda_context.cc DEPS gpu_info pten_device_context)
elseif (WITH_ROCM)
  #cc_library(rocm_context )
endif()

set(gpu_context cuda_context PARENT_SCOPE)

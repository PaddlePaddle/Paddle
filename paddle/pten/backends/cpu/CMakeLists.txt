if(WITH_MKLDNN)
  # TODO(wilber): support mkldnn context.
  cc_library(cpu_context SRCS cpu_context.cc DEPS pten_device_context mkldnn eigen)
else()
  cc_library(cpu_context SRCS cpu_context.cc DEPS pten_device_context eigen)
endif()

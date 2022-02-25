if(WITH_MKLDNN)
  # TODO(wilber): support mkldnn context.
  cc_library(cpu_context SRCS cpu_context.cc DEPS phi_device_context mkldnn eigen3)
else()
  cc_library(cpu_context SRCS cpu_context.cc DEPS phi_device_context eigen3)
endif()

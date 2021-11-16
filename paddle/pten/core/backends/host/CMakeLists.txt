if(WITH_ARM)
  #cc_library(arm_context SRCS arm_context.cc DEPS pten_device_context)
else()
  cc_library(x86_context SRCS x86_context.cc DEPS pten_device_context)
endif()

set(host_context x86_context PARENT_SCOPE)

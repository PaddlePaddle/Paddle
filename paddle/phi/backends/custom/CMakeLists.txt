if (WITH_CUSTOM_DEVICE)
  cc_library(custom_context SRCS custom_context.cc DEPS phi_device_context device_manager)
endif()

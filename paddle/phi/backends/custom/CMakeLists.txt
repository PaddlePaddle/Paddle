if(WITH_CUSTOM_DEVICE)
  cc_library(
    custom_context
    SRCS custom_context.cc
    DEPS phi_device_context phi_backends)
  cc_library(
    custom_device
    SRCS custom_device.cc
    DEPS device_base device_context)
  cc_test(
    custom_device_test
    SRCS custom_device_test.cc
    DEPS phi_backends device_context)
  cc_test(
    capi_test
    SRCS capi_test.cc
    DEPS phi_capi)
endif()

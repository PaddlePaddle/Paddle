IF(WITH_CUSTOM_DEVICE)
cc_library(custom_device SRCS custom_device.cc DEPS device_base)
cc_test(custom_device_test SRCS custom_device_test.cc DEPS device_manager device_context )
ENDIF()

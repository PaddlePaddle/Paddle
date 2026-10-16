if(NOT WIN32)
  cc_test(
    phi_test_device_manager
    SRCS test_device_manager.cc
    DEPS phi common)
endif()

if(WITH_CUSTOM_DEVICE)
  paddle_test(capi_test SRCS custom/capi_test.cc DEPS phi common)
endif()

if(WITH_CUSTOM_DEVICE)
  paddle_test(capi_test SRCS custom/capi_test.cc DEPS phi common)
endif()

if(NOT WIN32)
  paddle_test(dynamic_loader_test SRCS dynamic_loader_test.cc)
  target_compile_definitions(
    dynamic_loader_test
    PRIVATE DYNAMIC_LOADER_TEST_DSO_PATH="$<TARGET_FILE:common>")
endif()

if(WITH_CUSTOM_DEVICE)
  paddle_test(capi_test SRCS custom/capi_test.cc DEPS phi common)
endif()

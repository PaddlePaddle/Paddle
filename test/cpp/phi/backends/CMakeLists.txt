if(NOT WIN32)
  cc_library(dynamic_loader_test_dso SHARED SRCS dynamic_loader_test_dso.cc)
  paddle_test(dynamic_loader_test SRCS dynamic_loader_test.cc)
  add_dependencies(dynamic_loader_test dynamic_loader_test_dso)
  target_compile_definitions(
    dynamic_loader_test
    PRIVATE
      DYNAMIC_LOADER_TEST_DSO_PATH="$<TARGET_FILE:dynamic_loader_test_dso>")
endif()

if(WITH_CUSTOM_DEVICE)
  paddle_test(capi_test SRCS custom/capi_test.cc DEPS phi common)
endif()

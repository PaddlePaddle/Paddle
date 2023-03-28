include(ttest.cmake)
if(WITH_INFERENCE_API_TEST)
  add_subdirectory(api)
endif()

if(WITH_INFERENCE_API_TEST)
  include(paddle/fluid/inference/analysis/test.cmake)
  add_subdirectory(api)
endif()

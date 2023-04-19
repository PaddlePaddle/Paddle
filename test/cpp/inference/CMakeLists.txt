if(WITH_TESTING)
  include(test.cmake) # some generic cmake function for inference
endif()

add_subdirectory(analysis)

if(WITH_TESTING AND WITH_INFERENCE_API_TEST)
  add_subdirectory(api)
endif()

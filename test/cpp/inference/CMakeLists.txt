if(WITH_TESTING)
  include(test.cmake) # some generic cmake function for inference
endif()

add_subdirectory(analysis)
add_subdirectory(api)

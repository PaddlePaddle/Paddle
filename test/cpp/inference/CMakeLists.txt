add_definitions(-DPADDLE_DLL_EXPORT)
if(WITH_TESTING)
  include(test.cmake) # some generic cmake function for inference
  include(test_cases.cmake)
endif()

add_subdirectory(analysis)
add_subdirectory(api)
if(TENSORRT_FOUND AND WITH_TENSORRT)
  add_subdirectory(tensorrt)
endif()

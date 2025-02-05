file(
  GLOB TEST_MKLDNN_LISTS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_MKLDNN_LISTS "${TEST_MKLDNN_LISTS}")
if(WIN32)
  message(STATUS "Skip tests unrelated to onednn/mkldnn")
elseif(WITH_ONEDNN)
  foreach(target ${TEST_MKLDNN_LISTS})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER" TIMEOUT
                                              120)
  endforeach()
endif()

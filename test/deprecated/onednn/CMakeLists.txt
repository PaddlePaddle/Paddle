file(
  GLOB TEST_ONEDNN_LISTS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_ONEDNN_LISTS "${TEST_ONEDNN_LISTS}")
if(WIN32)
  message(STATUS "Skip tests unrelated to onednn")
elseif(WITH_ONEDNN)
  foreach(target ${TEST_ONEDNN_LISTS})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER" TIMEOUT
                                              120)
  endforeach()
endif()

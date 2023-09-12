if(WITH_IPU)
  file(
    GLOB CUSTOM_OP_TESTS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")
  string(REPLACE ".py" "" CUSTOM_OP_TESTS "${CUSTOM_OP_TESTS}")
  foreach(CUSTOM_OP_TEST ${CUSTOM_OP_TESTS})
    py_test(${CUSTOM_OP_TEST} SRCS ${CUSTOM_OP_TEST}.py)
  endforeach()

  add_subdirectory(deprecated)
endif()

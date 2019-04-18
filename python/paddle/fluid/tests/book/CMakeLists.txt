file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# default test
foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()

if(WITH_HIGH_LEVEL_API_TEST)
  add_subdirectory(high-level-api)
endif()

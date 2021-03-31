file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")


foreach(src ${TEST_OPS})
  py_test(${src} SRCS ${src}.py)
endforeach()

add_subdirectory(unittests)
add_subdirectory(book)

# 2.0 New custom OP can support Windows/Linux now
# TODO: support 2.0 New Custom OP on Mac
if(NOT APPLE)
  add_subdirectory(custom_op)
endif()

set_tests_properties(test_beam_search_decoder PROPERTIES TIMEOUT 120)

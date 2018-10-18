if(NOT APPLE)
  set(PYTHON_TESTS_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH "python tests directory")
else()
  set(PYTHON_TESTS_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests)
endif(NOT APPLE)

file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(src ${TEST_OPS})
  py_test(${src} SRCS ${src}.py)
endforeach()

add_subdirectory(unittests)
add_subdirectory(book)
add_subdirectory(book_memory_optimization)

file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
list(filter TEST_OPS EXCLUDE REGEX "test_\w+_op.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# merge tests for accelerating testing
set(MERGE_TESTS "merged_tests")
add_custom_command(DUMMY)
add_custom_command(TARGET DUMMY
  COMMAND python package_tests.py > ${MERGE_TESTS}.py)
py_test(${MERGE_TESTS} SRCS ${MERGE_TESTS}.py)

foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()

add_subdirectory(book)
add_subdirectory(book_distribute)
add_subdirectory(book_memory_optimization)

file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(src ${TEST_OPS})
  py_test(${src} SRCS ${src}.py)
endforeach()

add_subdirectory(unittests)
add_subdirectory(book)
add_subdirectory(custom_op)
add_subdirectory(custom_kernel)
add_subdirectory(custom_runtime)
<<<<<<< HEAD
=======

set_tests_properties(test_beam_search_decoder PROPERTIES TIMEOUT 120)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

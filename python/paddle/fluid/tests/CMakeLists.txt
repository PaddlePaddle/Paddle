file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(src ${TEST_OPS})
  py_test(${src} SRCS ${src}.py)
endforeach()

<<<<<<< HEAD
add_subdirectory(cpp_extension)
=======
add_subdirectory(unittests)
>>>>>>> e85fbac85fe3be0070a29ffa30c4c841de0c7a21

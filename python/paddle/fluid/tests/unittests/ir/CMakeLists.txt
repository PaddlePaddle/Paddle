file(GLOB TEST_IR_PASSES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_IR_PASSES "${TEST_IR_PASSES}")

foreach(target ${TEST_IR_PASSES})
  py_test_modules(${target} MODULES ${target})
endforeach()

add_subdirectory(inference)

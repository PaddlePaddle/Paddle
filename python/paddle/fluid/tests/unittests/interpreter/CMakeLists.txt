file(GLOB TEST_INTERP_CASES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_INTERP_CASES "${TEST_INTERP_CASES}")

list(REMOVE_ITEM TEST_INTERP_CASES test_standalone_executor)

foreach(target ${TEST_INTERP_CASES})
  py_test_modules(${target} MODULES ${target})
endforeach()

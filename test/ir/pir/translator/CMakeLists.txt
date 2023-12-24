file(
  GLOB TEST_INTERP_CASES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_INTERP_CASES "${TEST_INTERP_CASES}")

set(DISTRIBUTED_OP_TRANSLATION_TEST test_c_reduce_min_translate)

if(NOT WITH_DISTRIBUTE)
  list(REMOVE_ITEM TEST_INTERP_CASES ${DISTRIBUTED_OP_TRANSLATION_TEST})
endif()

foreach(target ${TEST_INTERP_CASES})
  py_test_modules(${target} MODULES ${target})
endforeach()

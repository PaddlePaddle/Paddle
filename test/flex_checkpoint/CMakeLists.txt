file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

set(GPU_ONLY_DISTRIBUTED_TESTS test_sharded_state_dict)

if(NOT (WITH_DISTRIBUTE AND WITH_GPU))
  foreach(t ${GPU_ONLY_DISTRIBUTED_TESTS})
    if(TEST ${t})
      set_property(TEST ${t} PROPERTY DISABLED TRUE)
    endif()
  endforeach()
endif()

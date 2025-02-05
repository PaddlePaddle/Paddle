file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
set(GC_ENVS FLAGS_eager_delete_tensor_gb=0.0)

if(WIN32)
  # TODO: Fix these unittests failed on Windows
  list(REMOVE_ITEM TEST_OPS test_autograd_functional_prim)
endif()

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach()

set_tests_properties(test_autograd_dynamic PROPERTIES TIMEOUT 100)
set_tests_properties(test_autograd_functional_dynamic PROPERTIES TIMEOUT 200)
if(NOT WIN32)
  set_tests_properties(test_autograd_functional_prim PROPERTIES TIMEOUT 60)
endif()
set_tests_properties(test_autograd_functional_static PROPERTIES TIMEOUT 160)

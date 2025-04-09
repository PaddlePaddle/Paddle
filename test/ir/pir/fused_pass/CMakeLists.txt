if(WITH_ONEDNN)
  add_subdirectory(onednn)
endif()

if(WITH_XPU)
  add_subdirectory(xpu)
endif()

if(WITH_PYTHON)
  add_subdirectory(python)
endif()

file(
  GLOB TEST_INTERP_CASES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_INTERP_CASES "${TEST_INTERP_CASES}")
if(NOT WITH_CUTLASS)
  set(CUTLASS_TEST_CASES test_fused_weight_only_linear_pass)
  list(REMOVE_ITEM TEST_INTERP_CASES ${CUTLASS_TEST_CASES})
endif()

foreach(target ${TEST_INTERP_CASES})
  py_test_modules(${target} MODULES ${target})
endforeach()

set_tests_properties(test_pir_multihead_matmul_fuse_pass PROPERTIES TIMEOUT 100)
set_tests_properties(test_add_norm_fuse_pass PROPERTIES TIMEOUT 300)
if(WITH_CUTLASS)
  set_tests_properties(test_fused_weight_only_linear_pass PROPERTIES TIMEOUT
                                                                     600)
endif()

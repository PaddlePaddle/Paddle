file(
  GLOB TEST_INTERP_CASES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_INTERP_CASES "${TEST_INTERP_CASES}")
if(NOT WITH_CUTLASS)
  set(CUTLASS_TEST_CASES test_fused_weight_only_linear_pass)
  list(REMOVE_ITEM TEST_INTERP_CASES ${CUTLASS_TEST_CASES})
endif()

if(WITH_COVERAGE)
  list(REMOVE_ITEM TEST_INTERP_CASES test_fused_gemm_epilogue_pass)
endif()

foreach(target ${TEST_INTERP_CASES})
  py_test_modules(${target} MODULES ${target})
endforeach()

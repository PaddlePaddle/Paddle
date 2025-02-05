file(
  GLOB TEST_INTERP_CASES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_INTERP_CASES "${TEST_INTERP_CASES}")
if((NOT WITH_MKLML)
   OR (NOT
       (WITH_AVX
        AND AVX512F_FOUND
        AND AVX512F_FLAG)
      ))
  set(SELF_ATTENTION_TEST_CASES test_self_attention_fuse_pass)
  list(REMOVE_ITEM TEST_INTERP_CASES ${SELF_ATTENTION_TEST_CASES})
endif()

foreach(target ${TEST_INTERP_CASES})
  py_test_modules(${target} MODULES ${target})
endforeach()

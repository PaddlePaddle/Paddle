file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if((NOT WITH_GPU) AND (NOT WITH_XPU))
  list(REMOVE_ITEM TEST_OPS "test_dist_fuse_all_reduce_pass")
endif()

if(NOT ((WITH_GPU) AND (CUDA_VERSION GREATER_EQUAL 11.6)))
  list(REMOVE_ITEM TEST_OPS test_dist_fuse_gemm_epilogue_pass)
endif()

if(NOT WITH_CUDNN_FRONTEND)
  list(REMOVE_ITEM TEST_OPS "test_dist_fuse_resunit_pass")
endif()

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS "NVIDIA_TF32_OVERRIDE=0")
  list(APPEND DIST_TEST_OPS ${TEST_OP})
  set_tests_properties(${TEST_OP} PROPERTIES TIMEOUT 250)
  set_tests_properties(${TEST_OP} PROPERTIES LABELS "RUN_TYPE=DIST")
endforeach()

if(WITH_CUDNN_FRONTEND)
  set_tests_properties(test_dist_fuse_resunit_pass PROPERTIES TIMEOUT 350)
endif()

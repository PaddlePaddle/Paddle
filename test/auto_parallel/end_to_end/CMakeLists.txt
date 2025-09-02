# file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
# string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(WITH_DISTRIBUTE AND WITH_GPU)
  if(LINUX)
    # test with eight cards
    py_test_modules(
      test_e2e_co_shard_8cards MODULES test_e2e_co_shard_8cards ENVS
      "http_proxy=;https_proxy=;PYTHONPATH=../..:${PADDLE_BINARY_DIR}/python")
    set_tests_properties(test_e2e_co_shard_8cards
                         PROPERTIES TIMEOUT "60" LABELS "RUN_TYPE=HYBRID")
  endif()
  py_test_modules(test_e2e_co_shard MODULES test_e2e_co_shard)

endif()

set_pir_tests_properties()

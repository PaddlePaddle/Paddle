if(WITH_DISTRIBUTE AND WITH_GPU)

  # NOTE(zyl): unittests WITH multi cards and timeout
  py_test_modules(test_semi_auto_parallel_custom_op MODULES
                  test_semi_auto_parallel_custom_op ENVS 
                  "http_proxy=;https_proxy=;PYTHONPATH=../..:${PADDLE_BINARY_DIR}/python")
  set_tests_properties(test_semi_auto_parallel_custom_op
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)

endif()

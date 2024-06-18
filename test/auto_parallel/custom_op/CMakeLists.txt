set(LOCAL_ALL_ARCH ON)
set(LOCAL_ALL_PLAT ON)
if(WITH_DISTRIBUTE
   AND WITH_GPU
   AND (LINUX))
  py_test_modules(
    test_semi_auto_parallel_custom_op
    MODULES
    test_semi_auto_parallel_custom_op
    ENVS
    "http_proxy=;https_proxy=;PYTHONPATH=../..:${PADDLE_BINARY_DIR}/python;PADDLE_SOURCE_DIR=${PROJECT_SOURCE_DIR};WITH_ONEDNN=${WITH_ONEDNN};ONEDNN_INSTALL_DIR=${ONEDNN_INSTALL_DIR};WITH_ONEDNN=${WITH_ONEDNN};WITH_GPU=${WITH_GPU};WITH_ROCM=${WITH_ROCM};externalError_INCLUDE_DIR=${externalError_INCLUDE_DIR};PYBIND_INCLUDE_DIR=${PYBIND_INCLUDE_DIR}"
  )
  set_tests_properties(test_semi_auto_parallel_custom_op
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)

endif()

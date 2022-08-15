set(LOCAL_ALL_ARCH ON)
set(LOCAL_ALL_PLAT ON)
if((WITH_GPU OR WITH_ROCM) AND (LINUX))
  py_test_modules(test_allgather MODULES test_allgather ENVS
                  "PADDLE_DIST_UT_PORT=20069;http_proxy=;https_proxy=")
  set_tests_properties(test_allgather PROPERTIES TIMEOUT "120" RUN_SERIAL 1)
endif()
if((WITH_GPU OR WITH_ROCM) AND (LINUX))
  py_test_modules(test_allreduce MODULES test_allreduce ENVS
                  "PADDLE_DIST_UT_PORT=20071;http_proxy=;https_proxy=")
  set_tests_properties(test_allreduce PROPERTIES TIMEOUT "120" RUN_SERIAL 1)
endif()
if((WITH_GPU OR WITH_ROCM) AND (LINUX))
  py_test_modules(test_broadcast MODULES test_broadcast ENVS
                  "PADDLE_DIST_UT_PORT=20073;http_proxy=;https_proxy=")
  set_tests_properties(test_broadcast PROPERTIES TIMEOUT "120" RUN_SERIAL 1)
endif()
if((WITH_GPU OR WITH_ROCM) AND (LINUX))
  py_test_modules(test_c_concat MODULES test_c_concat ENVS
                  "PADDLE_DIST_UT_PORT=20075;http_proxy=;https_proxy=")
  set_tests_properties(test_c_concat PROPERTIES TIMEOUT "120" RUN_SERIAL 1)
endif()
if((WITH_GPU OR WITH_ROCM) AND (LINUX))
  py_test_modules(test_c_identity MODULES test_c_identity ENVS
                  "PADDLE_DIST_UT_PORT=20077;http_proxy=;https_proxy=")
  set_tests_properties(test_c_identity PROPERTIES TIMEOUT "120" RUN_SERIAL 1)
endif()
if((WITH_GPU OR WITH_ROCM) AND (LINUX))
  py_test_modules(test_c_split MODULES test_c_split ENVS
                  "PADDLE_DIST_UT_PORT=20079;http_proxy=;https_proxy=")
  set_tests_properties(test_c_split PROPERTIES TIMEOUT "120" RUN_SERIAL 1)
endif()
if((WITH_GPU OR WITH_ROCM) AND (LINUX))
  py_test_modules(
    test_collective_allgather_api MODULES test_collective_allgather_api ENVS
    "PADDLE_DIST_UT_PORT=20081;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_allgather_api PROPERTIES TIMEOUT "300"
                                                                RUN_SERIAL 1)
endif()
if((WITH_GPU OR WITH_ROCM) AND (LINUX))
  py_test_modules(
    test_collective_allgather_object_api MODULES
    test_collective_allgather_object_api ENVS
    "PADDLE_DIST_UT_PORT=20083;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_allgather_object_api
                       PROPERTIES TIMEOUT "120" RUN_SERIAL 1)
endif()
if((WITH_GPU OR WITH_ROCM) AND (LINUX))
  py_test_modules(
    test_collective_allreduce_api MODULES test_collective_allreduce_api ENVS
    "PADDLE_DIST_UT_PORT=20085;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_allreduce_api PROPERTIES TIMEOUT "120"
                                                                RUN_SERIAL 1)
endif()

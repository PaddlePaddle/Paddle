set(LOCAL_ALL_ARCH ON)
set(LOCAL_ALL_PLAT ON)
if(LOCAL_ALL_ARCH AND (WIN32))
  py_test_modules(test_allgather MODULES test_allgather ENVS
                  "PADDLE_DIST_UT_PORT=20057;http_proxy=;https_proxy=")
  set_tests_properties(test_allgather PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND (WIN32))
  py_test_modules(test_allreduce MODULES test_allreduce ENVS
                  "PADDLE_DIST_UT_PORT=20059;http_proxy=;https_proxy=")
  set_tests_properties(test_allreduce PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND (WIN32))
  py_test_modules(test_broadcast MODULES test_broadcast ENVS
                  "PADDLE_DIST_UT_PORT=20061;http_proxy=;https_proxy=")
  set_tests_properties(test_broadcast PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND (WIN32))
  py_test_modules(test_c_concat MODULES test_c_concat ENVS
                  "PADDLE_DIST_UT_PORT=20063;http_proxy=;https_proxy=")
  set_tests_properties(test_c_concat PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND (WIN32))
  py_test_modules(test_c_identity MODULES test_c_identity ENVS
                  "PADDLE_DIST_UT_PORT=20065;http_proxy=;https_proxy=")
  set_tests_properties(test_c_identity PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND (WIN32))
  py_test_modules(test_c_split MODULES test_c_split ENVS
                  "PADDLE_DIST_UT_PORT=20067;http_proxy=;https_proxy=")
  set_tests_properties(test_c_split PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_allgather_api MODULES test_collective_allgather_api ENVS
    "PADDLE_DIST_UT_PORT=20081;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_allgather_api PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_allgather_object_api MODULES
    test_collective_allgather_object_api ENVS
    "PADDLE_DIST_UT_PORT=20083;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_allgather_object_api PROPERTIES TIMEOUT
                                                                       "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_allreduce_api MODULES test_collective_allreduce_api ENVS
    "PADDLE_DIST_UT_PORT=20085;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_allreduce_api PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_alltoall_api MODULES test_collective_alltoall_api ENVS
    "PADDLE_DIST_UT_PORT=20087;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_alltoall_api PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_collective_alltoall_single
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20089;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_alltoall_single PROPERTIES TIMEOUT "350")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_barrier_api MODULES test_collective_barrier_api ENVS
    "PADDLE_DIST_UT_PORT=20091;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_barrier_api PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_collective_batch_isend_irecv
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20093;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_batch_isend_irecv PROPERTIES TIMEOUT
                                                                    "350")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_broadcast_api MODULES test_collective_broadcast_api ENVS
    "PADDLE_DIST_UT_PORT=20095;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_broadcast_api PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_cpu_barrier_with_gloo MODULES
    test_collective_cpu_barrier_with_gloo ENVS
    "PADDLE_DIST_UT_PORT=20097;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_cpu_barrier_with_gloo PROPERTIES TIMEOUT
                                                                        "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_global_gather MODULES test_collective_global_gather ENVS
    "PADDLE_DIST_UT_PORT=20099;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_global_gather PROPERTIES TIMEOUT "200")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_global_scatter MODULES test_collective_global_scatter ENVS
    "PADDLE_DIST_UT_PORT=20101;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_global_scatter PROPERTIES TIMEOUT "200")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_collective_optimizer MODULES test_collective_optimizer
                  ENVS "PADDLE_DIST_UT_PORT=20103;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_optimizer PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_collective_process_group
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20105;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_process_group PROPERTIES TIMEOUT "350")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_collective_reduce MODULES test_collective_reduce ENVS
                  "PADDLE_DIST_UT_PORT=20107;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_reduce PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_collective_reduce_api MODULES test_collective_reduce_api
                  ENVS "PADDLE_DIST_UT_PORT=20109;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_reduce_api PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_collective_reduce_scatter
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20111;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_reduce_scatter PROPERTIES TIMEOUT "350")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_collective_scatter MODULES test_collective_scatter ENVS
                  "PADDLE_DIST_UT_PORT=20113;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_scatter PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_scatter_api MODULES test_collective_scatter_api ENVS
    "PADDLE_DIST_UT_PORT=20115;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_scatter_api PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_collective_sendrecv MODULES test_collective_sendrecv
                  ENVS "PADDLE_DIST_UT_PORT=20117;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_sendrecv PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_sendrecv_api MODULES test_collective_sendrecv_api ENVS
    "PADDLE_DIST_UT_PORT=20119;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_sendrecv_api PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_split_col_linear MODULES test_collective_split_col_linear
    ENVS "PADDLE_DIST_UT_PORT=20121;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_split_col_linear PROPERTIES TIMEOUT
                                                                   "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  bash_test_modules(
    test_collective_split_embedding
    START_BASH
    dist_test.sh
    LABELS
    "RUN_TYPE=DIST"
    ENVS
    "PADDLE_DIST_UT_PORT=20123;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_split_embedding PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_split_embedding_none_divisible MODULES
    test_collective_split_embedding_none_divisible ENVS
    "PADDLE_DIST_UT_PORT=20125;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_split_embedding_none_divisible
                       PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(
    test_collective_split_row_linear MODULES test_collective_split_row_linear
    ENVS "PADDLE_DIST_UT_PORT=20127;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_split_row_linear PROPERTIES TIMEOUT
                                                                   "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_collective_wait MODULES test_collective_wait ENVS
                  "PADDLE_DIST_UT_PORT=20129;http_proxy=;https_proxy=")
  set_tests_properties(test_collective_wait PROPERTIES TIMEOUT "300")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_reducescatter MODULES test_reducescatter ENVS
                  "PADDLE_DIST_UT_PORT=20131;http_proxy=;https_proxy=")
  set_tests_properties(test_reducescatter PROPERTIES TIMEOUT "120")
endif()
if(LOCAL_ALL_ARCH AND ((NOT APPLE AND NOT WIN32) OR WIN32))
  py_test_modules(test_reducescatter_api MODULES test_reducescatter_api ENVS
                  "PADDLE_DIST_UT_PORT=20133;http_proxy=;https_proxy=")
  set_tests_properties(test_reducescatter_api PROPERTIES TIMEOUT "120")
endif()

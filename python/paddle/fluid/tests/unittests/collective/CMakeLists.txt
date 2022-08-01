file(
  GLOB TEST_CASES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_CASES "${TEST_CASES}")
list(REMOVE_ITEM TEST_CASES test_collective_optimizer)
list(APPEND TEST_CASES test_eager_dist_api)

set(PY_TEST_CASES test_collective_optimizer test_new_group_api
                  test_eager_dist_api)

foreach(TEST ${PY_TEST_CASES})
  list(REMOVE_ITEM TEST_CASES ${TEST})
  py_test_modules(${TEST} MODULES ${TEST} ENVS ${dist_ENVS})
endforeach()

if(WITH_GPU)
  foreach(TEST_CASE ${TEST_CASES})
    bash_test_modules(
      ${TEST_CASE}
      START_BASH
      dist_test.sh
      LABELS
      "RUN_TYPE=EXCLUSIVE"
      ENVS
      "PADDLE_DIST_UT_PORT=${dist_ut_port}")
    # set_tests_properties(test_collective_alltoall_api
    #                     PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  endforeach()

endif()

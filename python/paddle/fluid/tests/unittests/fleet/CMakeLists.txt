file(
  GLOB TEST_CASES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_CASES "${TEST_CASES}")
list(REMOVE_ITEM TEST_CASES test_collective_optimizer)
if(WITH_GPU)
  foreach(TEST_CASE ${TEST_CASES})
    message("fleet test: " ${TEST_CASE})
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

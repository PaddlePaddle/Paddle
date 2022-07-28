file(
  GLOB TEST_CASES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_CASES "${TEST_CASES}")

function(set_py_test_module TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs ENVS TEST_CASES)
  cmake_parse_arguments(set_py_test_module "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  message("set py test module: " ${TARGET_NAME})

  list(REMOVE_ITEM set_py_test_module_TEST_CASES ${TARGET_NAME})

  py_test_modules(${TARGET_NAME} MODULES ${TARGET_NAME} ENVS
                  ${set_py_test_module_ENVS})
  set(TEST_CASES
      ${set_py_test_module_TEST_CASES}
      PARENT_SCOPE)
endfunction()

set(PY_TEST_CASES
    test_fleet_graph_executor
    test_fleet_rolemaker_3
    test_fleet_util
    test_fleet_utils
    test_fleet_meta_optimizer_base
    test_fleet
    test_fleet_base
    test_fleet_rolemaker
    test_fleet_amp_init
    test_fleet_checkpoint
    test_fleet_graph_execution_meta_optimizer
    test_fleet_nocvm_1
    test_fleet_rolemaker_new
    test_fleet_unitaccessor)
foreach(CASE ${PY_TEST_CASES})
  set_py_test_module(${CASE} TEST_CASES ${TEST_CASES} ENVS
                     "FLAGS_enable_eager_mode=1")
endforeach()

set(SINGLE_GPU_TESTS test_fleet_meta_optimizer_base test_fleet_amp_init
                     test_fleet_util)
foreach(CASE ${SINGLE_GPU_TESTS})
  set_tests_properties(
    ${CASE} PROPERTIES ENVIRONMENT
                       "CUDA_VISIBLE_DEVICES=0;FLAGS_enable_eager_mode=1")
endforeach()

list(REMOVE_ITEM TEST_CASES test_dist_mnist_train)

if(WITH_GPU)
  set(dist_ut_port 20001)
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
  endforeach()
endif()

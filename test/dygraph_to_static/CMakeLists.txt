file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
set(SOT_ENVS SOT_LOG_LEVEL=0 COST_MODEL=False MIN_GRAPH_SIZE=0
             STRICT_MODE=False)

if(WIN32 AND NOT WITH_GPU)
  # disable on Windows CPU CI for timeout
  list(REMOVE_ITEM TEST_OPS test_resnet_amp)
endif()

list(REMOVE_ITEM TEST_OPS test_build_strategy)

if(NOT WITH_GPU)
  # TODO(SigureMo): Temporarily disable train step on Windows CPU CI.
  # We should remove this after fix the performance issue.
  list(REMOVE_ITEM TEST_OPS test_train_step_resnet18_adam)
  list(REMOVE_ITEM TEST_OPS test_train_step_resnet18_sgd)
  # disable some model test on CPU to avoid timeout
  list(REMOVE_ITEM TEST_OPS test_resnet)
  list(REMOVE_ITEM TEST_OPS test_bert)
  list(REMOVE_ITEM TEST_OPS test_transformer)
  list(REMOVE_ITEM TEST_OPS test_mobile_net)
  list(REMOVE_ITEM TEST_OPS test_seq2seq)
endif()

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${SOT_ENVS})
endforeach()

set_tests_properties(test_se_resnet PROPERTIES TIMEOUT 900)
set_tests_properties(test_yolov3 PROPERTIES TIMEOUT 900 LABELS
                                            "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_cycle_gan PROPERTIES TIMEOUT 150)
set_tests_properties(test_basic_api_transformation PROPERTIES TIMEOUT 240)
set_tests_properties(test_reinforcement_learning PROPERTIES TIMEOUT 120)
set_tests_properties(test_bmn PROPERTIES TIMEOUT 300)
set_tests_properties(test_loop PROPERTIES TIMEOUT 180)
set_tests_properties(test_mnist_amp PROPERTIES TIMEOUT 240)
set_tests_properties(test_sentiment PROPERTIES TIMEOUT 150)
set_tests_properties(test_incubate_jit_inference PROPERTIES TIMEOUT 150)

if(TEST test_resnet_amp)
  set_tests_properties(test_resnet_amp PROPERTIES TIMEOUT 360)
endif()

if(NOT WIN32)
  set_tests_properties(test_tsm PROPERTIES TIMEOUT 900)
endif()

if(APPLE)
  set_tests_properties(test_bmn PROPERTIES TIMEOUT 300)
endif()

if(WITH_GPU)
  set_tests_properties(test_train_step_resnet18_sgd PROPERTIES TIMEOUT 240)
  set_tests_properties(test_train_step_resnet18_adam PROPERTIES TIMEOUT 240)
  set_tests_properties(test_bert PROPERTIES TIMEOUT 240)
  set_tests_properties(test_transformer PROPERTIES TIMEOUT 240)
  set_tests_properties(test_mobile_net PROPERTIES TIMEOUT 240)
  set_tests_properties(test_resnet PROPERTIES TIMEOUT 240)
  set_tests_properties(test_seq2seq PROPERTIES TIMEOUT 900)
endif()

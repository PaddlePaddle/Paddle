file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
set(SOT_ENVS SOT_LOG_LEVEL=0 COST_MODEL=False MIN_GRAPH_SIZE=0
             STRICT_MODE=False)
set(GC_ENVS FLAGS_eager_delete_tensor_gb=0.0)

list(REMOVE_ITEM TEST_OPS test_lac)
# NOTE(Aurelius84): In case of Windows CI, if open ON_INFER, RWLOCK of Scope
# will be removed and will cause some random failed in multi-thread.
if(WITH_PYTHON)
  py_test_modules(test_lac MODULES test_lac)
  set_tests_properties(test_lac PROPERTIES TIMEOUT 120)
endif()

if(WIN32 AND NOT WITH_GPU)
  list(REMOVE_ITEM TEST_OPS test_resnet_amp
  )# disable on Windows CPU CI for timeout
endif()

if(NOT WITH_GPU)
  # TODO(SigureMo): Temporarily disable train step on Windows CPU CI.
  # We should remove this after fix the performance issue.
  list(REMOVE_ITEM TEST_OPS test_train_step_resnet18_adam)
  list(REMOVE_ITEM TEST_OPS test_train_step_resnet18_sgd)
endif()

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS} ${SOT_ENVS})
endforeach()

set_tests_properties(test_se_resnet PROPERTIES TIMEOUT 900)
set_tests_properties(test_yolov3 PROPERTIES TIMEOUT 900 LABELS
                                            "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_mobile_net PROPERTIES TIMEOUT 120)
set_tests_properties(test_seq2seq PROPERTIES TIMEOUT 420)
set_tests_properties(test_cycle_gan PROPERTIES TIMEOUT 150)
set_tests_properties(test_bert PROPERTIES TIMEOUT 180)
set_tests_properties(test_basic_api_transformation PROPERTIES TIMEOUT 240)
set_tests_properties(test_reinforcement_learning PROPERTIES TIMEOUT 120)
set_tests_properties(test_transformer PROPERTIES TIMEOUT 200)
set_tests_properties(test_bmn PROPERTIES TIMEOUT 300)
set_tests_properties(test_bert PROPERTIES TIMEOUT 240)
#set_tests_properties(test_mnist PROPERTIES TIMEOUT 120)
set_tests_properties(test_build_strategy PROPERTIES TIMEOUT 120)

if(NOT WIN32)
  set_tests_properties(test_resnet_v2 PROPERTIES TIMEOUT 180)
  set_tests_properties(test_tsm PROPERTIES TIMEOUT 900)
  set_tests_properties(test_resnet PROPERTIES TIMEOUT 300)
endif()

if(APPLE)
  set_tests_properties(test_bmn PROPERTIES TIMEOUT 300)
  set_tests_properties(test_build_strategy PROPERTIES TIMEOUT 300)
  set_tests_properties(test_mobile_net PROPERTIES TIMEOUT 300)
  set_tests_properties(test_resnet_v2 PROPERTIES TIMEOUT 300)
endif()

if(WITH_GPU)
  set_tests_properties(test_train_step_resnet18_sgd PROPERTIES TIMEOUT 240)
  set_tests_properties(test_train_step_resnet18_adam PROPERTIES TIMEOUT 240)
endif()

set(DISABLE_PIR_PT_MODES
    test_error
    test_gradname_parse
    test_seq2seq
    test_pylayer
    test_save_inference_model
    test_tensor_hook
    test_len
    test_list
    test_slice
    test_lstm
    test_for_enumerate
    test_reinforcement_learning)
foreach(ITEST ${DISABLE_PIR_PT_MODES})
  if(TEST ${ITEST})
    set_tests_properties(
      ${ITEST} PROPERTIES ENVIRONMENT "FLAGS_enable_pir_with_pt_in_dy2st=0")
    message(
      STATUS
        "PIR Disabled OpTest: set FLAGS_enable_pir_with_pt_in_dy2st to False for ${ITEST}"
    )
  else()
    message(
      STATUS "PIR Disabled OpTest: not found ${ITEST} in dygraph_to_static")
  endif()
endforeach()

if(NOT WIN32)
  set(WIN_DISABLE_PIR_PT_MODES test_jit_setitem test_program_translator
                               test_cache_program)
  foreach(ITEST ${WIN_DISABLE_PIR_PT_MODES})
    if(TEST ${ITEST})
      set_tests_properties(
        ${ITEST} PROPERTIES ENVIRONMENT "FLAGS_enable_pir_with_pt_in_dy2st=0")
      message(
        STATUS
          "WIN PIR Disabled OpTest: set FLAGS_enable_pir_with_pt_in_dy2st to False for ${ITEST}"
      )
    else()
      message(
        STATUS
          "WIN PIR Disabled OpTest: not found ${ITEST} in dygraph_to_static")
    endif()
  endforeach()
endif()

file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
set(GC_ENVS FLAGS_eager_delete_tensor_gb=0.0)

list(REMOVE_ITEM TEST_OPS test_lac)
# NOTE(Aurelius84): In case of Windows CI, if open ON_INFER, RWLOCK of Scope will
# be removed and will cause some random failed in multi-thread.
if(NOT ON_INFER)
    py_test_modules(test_lac MODULES test_lac)
    set_tests_properties(test_lac PROPERTIES TIMEOUT 120)
endif()

if(WIN32 AND NOT WITH_GPU)
    list(REMOVE_ITEM TEST_OPS test_resnet_amp) # disable on Windows CPU CI for timeout
endif()

foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach(TEST_OP)

set_tests_properties(test_se_resnet PROPERTIES TIMEOUT 900)
set_tests_properties(test_yolov3 PROPERTIES TIMEOUT 900 LABELS "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_mobile_net PROPERTIES TIMEOUT 120)
set_tests_properties(test_seq2seq PROPERTIES TIMEOUT 120)
set_tests_properties(test_cycle_gan PROPERTIES TIMEOUT 150)
set_tests_properties(test_bert PROPERTIES TIMEOUT 120)
set_tests_properties(test_basic_api_transformation PROPERTIES TIMEOUT 120)
set_tests_properties(test_reinforcement_learning PROPERTIES TIMEOUT 120)
set_tests_properties(test_transformer PROPERTIES TIMEOUT 200)
set_tests_properties(test_bmn PROPERTIES TIMEOUT 120)
#set_tests_properties(test_mnist PROPERTIES TIMEOUT 120)
set_tests_properties(test_build_strategy PROPERTIES TIMEOUT 120)

if(NOT WIN32)
    set_tests_properties(test_resnet_v2 PROPERTIES TIMEOUT 120)
    set_tests_properties(test_tsm PROPERTIES TIMEOUT 900)
    #set_tests_properties(test_resnet PROPERTIES TIMEOUT 120)
endif()

if(APPLE)
    set_tests_properties(test_bmn PROPERTIES TIMEOUT 300)
    set_tests_properties(test_build_strategy PROPERTIES TIMEOUT 300)
    set_tests_properties(test_mobile_net PROPERTIES TIMEOUT 300)
    set_tests_properties(test_resnet_v2 PROPERTIES TIMEOUT 300)
endif()

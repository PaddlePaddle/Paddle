file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
set(GC_ENVS FLAGS_eager_delete_tensor_gb=0.0)
set(DY2ST_EAGER_TEST_ENVS ${GC_ENVS} FLAGS_enable_eager_mode=1)

set(TEST_EAGER_OPS test_bmn test_break_continue test_ifelse test_loop test_mnist_amp 
    test_mnist_pure_fp16 test_mobile_net test_program_translator test_ptb_lm test_reinforcement_learning 
    test_resnet test_resnet_amp test_resnet_pure_fp16 test_se_resnet test_sentiment test_seq2seq 
    test_tsm test_word2vec test_yolov3 test_bert test_cycle_gan test_lstm test_simnet test_transformer)
list(REMOVE_ITEM TEST_OPS test_lac)
# NOTE(Aurelius84): In case of Windows CI, if open ON_INFER, RWLOCK of Scope will
# be removed and will cause some random failed in multi-thread.
if(NOT ON_INFER)
    py_test_modules(test_lac MODULES test_lac ENVS FLAGS_enable_eager_mode=1)
    set_tests_properties(test_lac PROPERTIES TIMEOUT 120)
endif()

if(WIN32 AND NOT WITH_GPU)
    list(REMOVE_ITEM TEST_OPS test_resnet_amp) # disable on Windows CPU CI for timeout
endif()

foreach(TEST_OP ${TEST_OPS})
    list(FIND TEST_EAGER_OPS ${TEST_OP} WAS_FOUND)
    if (NOT WAS_FOUND EQUAL -1)
        py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${DY2ST_EAGER_TEST_ENVS})
    else()
        py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
    endif()
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

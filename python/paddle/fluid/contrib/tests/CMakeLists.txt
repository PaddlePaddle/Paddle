file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

list(REMOVE_ITEM TEST_OPS test_multi_precision_fp16_train)
list(REMOVE_ITEM TEST_OPS test_fit_a_line_bf16)
list(REMOVE_ITEM TEST_OPS test_model_cast_to_bf16)

foreach(src ${TEST_OPS})
        py_test(${src} SRCS ${src}.py)
endforeach()

py_test_modules(test_multi_precision_fp16_train MODULES test_multi_precision_fp16_train ENVS FLAGS_cudnn_deterministic=true FLAGS_cudnn_batchnorm_spatial_persistent=true FLAGS_conv_workspace_size_limit=1000)

set_tests_properties(test_image_classification_fp16 PROPERTIES TIMEOUT 120)
set_tests_properties(test_weight_decay_extend PROPERTIES TIMEOUT 120)
set_tests_properties(test_multi_precision_fp16_train PROPERTIES TIMEOUT 120)

if(WITH_MKLDNN)
    py_test_modules(test_fit_a_line_bf16 MODULES test_fit_a_line_bf16)
    py_test_modules(test_model_cast_to_bf16 MODULES test_model_cast_to_bf16)
    set_tests_properties(test_model_cast_to_bf16 PROPERTIES TIMEOUT 120)
    set_tests_properties(test_fit_a_line_bf16 PROPERTIES TIMEOUT 120)
endif()

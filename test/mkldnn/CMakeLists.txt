file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

list(REMOVE_ITEM TEST_OPS "test_onnx_format_quantization_mobilenetv1")
list(REMOVE_ITEM TEST_OPS "test_flags_mkldnn_ops_on_off")

if(WITH_MKLDNN AND NOT WIN32)
  list(APPEND TEST_OPS "test_onnx_format_quantization_mobilenetv1")
endif()

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

# NODE(Ruibiao): Remove it after static build is enabled by default.
if(WITH_MKLDNN AND NOT WIN32)
  py_test_modules(
    test_dequantize_mkldnn_op_static_build MODULES test_dequantize_mkldnn_op
    ENVS FLAGS_new_executor_static_build=true)
endif()

set_tests_properties(test_concat_mkldnn_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv3d_mkldnn_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_mul_onednn_op PROPERTIES TIMEOUT 60)
set_tests_properties(test_elementwise_add_mkldnn_op PROPERTIES TIMEOUT 60)
if(WITH_MKLDNN AND NOT WIN32)
  set_tests_properties(test_onnx_format_quantization_mobilenetv1
                       PROPERTIES TIMEOUT 300)
endif()
# set_tests_properties(test_flags_mkldnn_ops_on_off PROPERTIES TIMEOUT 120)

set_pit_tests_properties()

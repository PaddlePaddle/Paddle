file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

list(REMOVE_ITEM TEST_OPS "test_onnx_format_quantization_mobilenetv1")
<<<<<<< HEAD
list(REMOVE_ITEM TEST_OPS "test_flags_mkldnn_ops_on_off")
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if(WITH_MKLDNN AND NOT WIN32)
  list(APPEND TEST_OPS "test_onnx_format_quantization_mobilenetv1")
endif()

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()
set_tests_properties(test_concat_mkldnn_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv3d_mkldnn_op PROPERTIES TIMEOUT 120)
if(WITH_MKLDNN AND NOT WIN32)
  set_tests_properties(test_onnx_format_quantization_mobilenetv1
                       PROPERTIES TIMEOUT 300)
endif()
<<<<<<< HEAD
# set_tests_properties(test_flags_mkldnn_ops_on_off PROPERTIES TIMEOUT 120)
=======
set_tests_properties(test_flags_mkldnn_ops_on_off PROPERTIES TIMEOUT 120)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

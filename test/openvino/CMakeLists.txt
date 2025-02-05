file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

py_test_modules(test_openvino_resnet50 MODULES test_openvino_resnet50 ENVS
                "FLAGS_enable_pir_api=0")
set_tests_properties(test_openvino_resnet50 PROPERTIES TIMEOUT 120)

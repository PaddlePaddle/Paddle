paddle_test(
  shape_op_test
  SRCS
  shape_op_test.cc
  DEPS
  pd_op_dialect
  pir
  gtest)

paddle_test(
  shape_struct_test
  SRCS
  shape_struct_test.cc
  DEPS
  pd_op_dialect
  pir
  gtest)

paddle_test(
  shape_optimization_test
  SRCS
  shape_optimization_test.cc
  DEPS
  gtest
  pd_op_dialect
  pir)

set_tests_properties(
  shape_optimization_test PROPERTIES ENVIRONMENT
                                     "FLAGS_enable_new_ir_in_executor=true")

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(shape_op_test)
endif()

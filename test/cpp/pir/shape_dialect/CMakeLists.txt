paddle_test(
  shape_op_test
  SRCS
  shape_op_test.cc
  DEPS
  op_dialect_vjp
  pir
  gtest)

paddle_test(
  shape_struct_test
  SRCS
  shape_struct_test.cc
  DEPS
  op_dialect_vjp
  pir
  gtest)

if(WITH_CINN)
  paddle_test(
    shape_optimization_test
    SRCS
    shape_optimization_test.cc
    DEPS
    gtest
    op_dialect_vjp
    pir
    pir_transforms)
  set_tests_properties(
    shape_optimization_test PROPERTIES ENVIRONMENT
                                       "FLAGS_enable_pir_in_executor=true")
endif()

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(shape_op_test)
endif()

if(NOT WIN32)
  nv_test(
    test_tensorrt_engine_op
    SRCS tensorrt_engine_op_test.cc
    DEPS tensorrt_engine_op analysis python)
else()
  get_property(paddle_lib GLOBAL PROPERTY PADDLE_LIB_NAME)
  nv_test(
    test_tensorrt_engine_op
    SRCS tensorrt_engine_op_test.cc
    DEPS tensorrt_engine_op analysis)
endif()

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_tensorrt_engine_op)
endif()

if(WIN32)
  nv_test(
    test_tensorrt
    SRCS test_tensorrt.cc
    DEPS phi common dynload_tensorrt)
else()
  nv_test(
    test_tensorrt
    SRCS test_tensorrt.cc
    DEPS phi common)
endif()
if(WIN32)
  nv_test(
    test_tensorrt_engine
    SRCS test_engine.cc test_dynamic_engine.cc
    DEPS phi common dynload_tensorrt tensorrt_engine tensorrt_plugin)
elseif(WITH_CINN)
  nv_test(
    test_tensorrt_engine
    SRCS test_engine.cc test_dynamic_engine.cc
    DEPS phi common tensorrt_engine tensorrt_plugin python)
else()
  nv_test(
    test_tensorrt_engine
    SRCS test_engine.cc test_dynamic_engine.cc
    DEPS phi common tensorrt_engine tensorrt_plugin python)
endif()
nv_test(
  test_arg_mapping_context
  SRCS test_arg_mapping_context.cc
  DEPS phi tensorrt_plugin_arg_mapping_context)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_tensorrt_engine)
endif()

add_subdirectory(plugin)

# Compiling with WITH_PYTHON=ON and WITH_TENSORRT=ON failed on windows.
# Temporarily add paddle_inference_api dependency to solve the problem
if(WIN32)
  nv_library(
    tensorrt_engine
    SRCS engine.cc trt_int8_calibrator.cc
    DEPS ${GLOB_OPERATOR_DEPS} framework_proto device_context boost
         paddle_inference_api)
else()
  nv_library(
    tensorrt_engine
    SRCS engine.cc trt_int8_calibrator.cc
    DEPS ${GLOB_OPERATOR_DEPS} framework_proto device_context boost)
endif()
nv_library(
  tensorrt_op_teller
  SRCS op_teller.cc
  DEPS framework_proto device_context boost)
nv_test(
  test_tensorrt
  SRCS test_tensorrt.cc
  DEPS dynload_cuda device_context dynamic_loader)
nv_test(
  test_tensorrt_engine
  SRCS test_engine.cc test_dynamic_engine.cc
  DEPS dynload_cuda tensorrt_engine tensorrt_plugin)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_tensorrt_engine)
endif()

add_subdirectory(plugin)
add_subdirectory(convert)

# Compiling with WITH_PYTHON=ON and WITH_TENSORRT=ON failed on windows.
# Temporarily add paddle_inference_api dependency to solve the problem
if(WIN32)
  nv_library(
    tensorrt_engine
    SRCS engine.cc trt_int8_calibrator.cc
    DEPS ${GLOB_OPERATOR_DEPS} phi paddle_inference_api)
else()
  nv_library(
    tensorrt_engine
    SRCS engine.cc trt_int8_calibrator.cc
    DEPS ${GLOB_OPERATOR_DEPS} phi)
endif()
nv_library(
  tensorrt_dynamic_shape_infermeta_factory
  SRCS dynamic_shape_infermeta.cc
  DEPS phi)
nv_library(
  tensorrt_plugin_arg_mapping_context
  SRCS plugin_arg_mapping_context.cc
  DEPS phi)
nv_library(
  tensorrt_op_teller
  SRCS op_teller.cc
  DEPS phi tensorrt_dynamic_shape_infermeta_factory)

add_subdirectory(plugin)
add_subdirectory(convert)
add_subdirectory(pir)

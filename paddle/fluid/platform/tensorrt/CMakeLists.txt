# Compiling with WITH_PYTHON=ON and WITH_TENSORRT=ON failed on windows.
# Temporarily add paddle_inference_api dependency to solve the problem
if(WIN32)
  nv_library(
    trt_engine
    SRCS engine.cc
    DEPS ${GLOB_OPERATOR_DEPS} framework_proto device_context
         paddle_inference_api)
else()
  nv_library(
    trt_engine
    SRCS engine.cc
    DEPS ${GLOB_OPERATOR_DEPS} framework_proto device_context)
endif()

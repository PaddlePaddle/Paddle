nv_library(
  trt_engine
  SRCS engine.cc
  DEPS device_context)

nv_library(
  trt_plugin
  SRCS trt_plugin.cc
  DEPS common)

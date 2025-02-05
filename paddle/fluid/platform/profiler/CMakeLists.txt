cc_library(
  host_tracer
  SRCS host_tracer.cc
  DEPS phi common var_type_traits)
cc_library(
  cuda_tracer
  SRCS cuda_tracer.cc cupti_data_process.cc
  DEPS standalone_executor phi glog common)
cc_library(
  xpu_tracer
  SRCS xpu_tracer.cc
  DEPS phi glog common)
add_subdirectory(custom_device)
cc_library(
  event_node
  SRCS event_node.cc
  DEPS phi common)
add_subdirectory(dump)
cc_library(
  profiler_logger
  SRCS chrometracing_logger.cc dump/serialization_logger.cc
       dump/deserialization_reader.cc
  DEPS nodetreeproto event_node phi glog common)
cc_library(
  event_bind
  SRCS event_python.cc
  DEPS profiler_logger)
cc_library(
  new_profiler
  SRCS profiler.cc
  DEPS host_tracer
       cuda_tracer
       xpu_tracer
       phi
       common
       glog
       event_bind
       custom_tracer)

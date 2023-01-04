cc_library(
  host_tracer
  SRCS host_tracer.cc
  DEPS framework_proto enforce ddim var_type_traits)
cc_library(
  cuda_tracer
  SRCS cuda_tracer.cc cupti_data_process.cc
  DEPS workqueue_utils enforce glog)
add_subdirectory(mlu)
add_subdirectory(custom_device)
cc_library(
  event_node
  SRCS event_node.cc
  DEPS enforce place)
cc_library(
  profiler_utils
  SRCS utils.cc
  DEPS enforce glog)
add_subdirectory(dump)
cc_library(
  profiler_logger
  SRCS chrometracing_logger.cc dump/serialization_logger.cc
       dump/deserialization_reader.cc
  DEPS nodetreeproto event_node profiler_utils)
cc_library(
  event_bind
  SRCS event_python.cc
  DEPS profiler_logger)
cc_library(
  cpu_utilization
  SRCS cpu_utilization.cc
  DEPS phi_backends os_info enforce glog)
cc_library(
  new_profiler
  SRCS profiler.cc
  DEPS host_tracer
       cuda_tracer
       profiler_utils
       cpu_utilization
       event_bind
       mlu_tracer
       custom_tracer)
cc_test(
  test_event_node
  SRCS test_event_node.cc
  DEPS event_node profiler_logger)
cc_test(
  test_extra_info
  SRCS test_extra_info.cc
  DEPS profiler_utils)
cc_test(
  test_serialization_logger
  SRCS dump/test_serialization_logger.cc
  DEPS event_bind)
cc_test(
  new_profiler_test
  SRCS profiler_test.cc
  DEPS new_profiler)

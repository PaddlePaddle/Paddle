add_subdirectory(dump)
cc_test(
  test_event_node
  SRCS test_event_node.cc
  DEPS event_node profiler_logger)
cc_test(
  test_extra_info
  SRCS test_extra_info.cc
  DEPS profiler_utils)
cc_test(
  new_profiler_test
  SRCS profiler_test.cc
  DEPS new_profiler)

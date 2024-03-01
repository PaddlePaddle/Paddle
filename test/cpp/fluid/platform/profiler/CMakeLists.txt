add_subdirectory(dump)
paddle_test(test_event_node SRCS test_event_node.cc)
paddle_test(test_extra_info SRCS test_extra_info.cc)
nv_test(
  new_profiler_test
  SRCS profiler_test.cc
  DEPS new_profiler)

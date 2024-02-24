add_subdirectory(device)
add_subdirectory(profiler)

paddle_test(errors_test SRCS errors_test.cc)

cc_test(
  enforce_test
  SRCS enforce_test.cc
  DEPS enforce common)

cc_test(
  cpu_info_test
  SRCS cpu_info_test.cc
  DEPS phi common)
cc_test(
  os_info_test
  SRCS os_info_test.cc
  DEPS phi common)

cc_test(
  place_test
  SRCS place_test.cc
  DEPS place glog phi common)

cc_test(
  cpu_helper_test
  SRCS cpu_helper_test.cc
  DEPS cpu_helper)

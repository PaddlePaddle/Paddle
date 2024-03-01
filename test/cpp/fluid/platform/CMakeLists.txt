add_subdirectory(device)
add_subdirectory(profiler)

paddle_test(errors_test SRCS errors_test.cc)

paddle_test(enforce_test SRCS enforce_test.cc DEPS enforce common external_error_proto)

paddle_test(cpu_info_test SRCS cpu_info_test.cc DEPS common)
paddle_test(os_info_test SRCS os_info_test.cc DEPS common)

cc_test(
  place_test
  SRCS place_test.cc
  DEPS place glog phi common)

cc_test(
  cpu_helper_test
  SRCS cpu_helper_test.cc
  DEPS cpu_helper)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(errors_test)
endif()

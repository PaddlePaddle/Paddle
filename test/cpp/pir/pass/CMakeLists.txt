paddle_test(pass_manager_test SRCS pass_manager_test.cc DEPS common)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(pass_manager_test)
endif()

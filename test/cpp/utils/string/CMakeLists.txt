paddle_test(stringprintf_test SRCS printf_test.cc)
paddle_test(to_string_test SRCS to_string_test.cc)
paddle_test(split_test SRCS split_test.cc)
paddle_test(string_helper_test SRCS string_helper_test.cc DEPS string_helper)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(stringprintf_test)
  copy_onnx(to_string_test)
  copy_onnx(split_test)
  copy_onnx(string_helper_test)
endif()

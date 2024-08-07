paddle_test(test_reference_count_pass_last_lived_ops SRCS
            test_reference_count_pass_last_lived_ops.cc DEPS common)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_reference_count_pass_last_lived_ops)
endif()

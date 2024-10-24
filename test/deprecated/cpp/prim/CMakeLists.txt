paddle_test(test_static_prim_deprecated SRCS test_static_prim_deprecated.cc)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_static_prim_deprecated)
endif()

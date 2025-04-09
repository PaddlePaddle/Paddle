if(WITH_CUSTOM_DEVICE)
  paddle_test(test_custom_engine_loadlib SRCS test_custom_engine_loadlib.cc
              custom_engine_op.cc)

endif()
paddle_test(test_custom_engine SRCS test_custom_engine_operation.cc
            custom_engine_op.cc)
if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_custom_engine)
endif()

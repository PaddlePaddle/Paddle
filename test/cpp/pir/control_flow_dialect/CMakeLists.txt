if(NOT WITH_ROCM)
  paddle_test(if_op_test SRCS if_op_test.cc)
  if(WITH_ONNXRUNTIME AND WIN32)
    # Copy onnxruntime for some c++ test in Windows, since the test will
    # be build only in CI, so suppose the generator in Windows is Ninja.
    copy_onnx(if_op_test)
  endif()

  paddle_test(while_op_test SRCS while_op_test.cc)
endif()

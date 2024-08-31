paddle_test(test_builtin_parameter SRCS test_builtin_parameter.cc)
paddle_test(save_load_version_compat_test SRCS save_load_version_compat_test.cc
            DEPS test_dialect)

copy_if_different(${CMAKE_CURRENT_SOURCE_DIR}/patch
                  ${CMAKE_CURRENT_BINARY_DIR}/patch)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_builtin_parameter)
endif()

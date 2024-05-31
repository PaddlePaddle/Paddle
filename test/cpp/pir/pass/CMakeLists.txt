paddle_test(pass_manager_test SRCS pass_manager_test.cc DEPS common)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(pass_manager_test)
endif()

if(WITH_GPU)
  file(DOWNLOAD https://paddle-ci.gz.bcebos.com/test/sd15_unet.pdmodel
       ${CMAKE_CURRENT_BINARY_DIR}/sd15_unet.pdmodel
       EXPECTED_MD5 4b5a3b8ea5b49bfd12172847cfe5a92a)

  paddle_test(transfer_layout_pass_test SRCS transfer_layout_pass_test.cc)
  if(WITH_ONNXRUNTIME AND WIN32)
    # Copy onnxruntime for some c++ test in Windows, since the test will
    # be build only in CI, so suppose the generator in Windows is Ninja.
    copy_onnx(transfer_layout_pass_test)
  endif()
endif()

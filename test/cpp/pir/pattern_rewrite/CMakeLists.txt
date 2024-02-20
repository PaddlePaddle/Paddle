paddle_test(pattern_rewrite_test SRCS pattern_rewrite_test.cc)

paddle_test(drr_same_type_binding_test SRCS drr_same_type_binding_test.cc)

paddle_test(drr_fuse_linear_test SRCS drr_fuse_linear_test.cc)

paddle_test(drr_fuse_linear_param_grad_add_test SRCS
            drr_fuse_linear_param_grad_add_test.cc)

if(WITH_GPU)
  paddle_test(drr_attention_fuse_test SRCS drr_attention_fuse_test.cc)
endif()

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(pattern_rewrite_test)
  copy_onnx(drr_same_type_binding_test)
  copy_onnx(drr_fuse_linear_test)
  copy_onnx(drr_fuse_linear_param_grad_add_test)
  if(WITH_GPU)
    copy_onnx(drr_attention_fuse_test)
  endif()
endif()

paddle_test(test_egr_task_nan_inf_utils SRCS nan_inf_utils_test.cc DEPS common)

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  paddle_test(test_egr_task_hook SRCS hook_test.cc)
  paddle_test(test_egr_task_backward SRCS backward_test.cc)
  paddle_test(test_egr_task_grad SRCS grad_test.cc)
  paddle_test(test_egr_task_fwd_bwd_joint SRCS fwd_bwd_joint_test.cc DEPS phi)
  paddle_test(test_egr_task_cross_batch SRCS cross_batch_accumulation_test.cc)
  paddle_test(test_egr_task_hook_intermidiate SRCS hook_test_intermidiate.cc)
  paddle_test(test_egr_task_autocodegen SRCS generated_test.cc)
  paddle_test(test_egr_task_tensor_utils SRCS tensor_utils_test.cc)
  paddle_test(test_egr_task_eager_utils SRCS eager_utils_test.cc)
  paddle_test(test_egr_task_forward_autograd SRCS forward_autograd_test.cc)
endif()

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_egr_task_nan_inf_utils)
endif()

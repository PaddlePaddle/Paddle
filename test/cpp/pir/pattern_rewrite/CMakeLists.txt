cc_test(
  pattern_rewrite_test
  SRCS pattern_rewrite_test.cc
  DEPS gtest op_dialect_vjp pir pir_transforms)

cc_test(
  drr_test
  SRCS drr_test.cc
  DEPS drr pir_transforms)

cc_test(
  drr_same_type_binding_test
  SRCS drr_same_type_binding_test.cc
  DEPS drr gtest op_dialect_vjp pir pir_transforms)

cc_test(
  drr_fuse_linear_test
  SRCS drr_fuse_linear_test.cc
  DEPS pir_transforms drr gtest op_dialect_vjp pir)

cc_test(
  drr_fuse_linear_param_grad_add_test
  SRCS drr_fuse_linear_param_grad_add_test.cc
  DEPS pir_transforms drr gtest op_dialect_vjp pir)

if(WITH_GPU)
  cc_test(
    drr_attention_fuse_test
    SRCS drr_attention_fuse_test.cc
    DEPS pir_transforms drr gtest op_dialect_vjp pir)
endif()

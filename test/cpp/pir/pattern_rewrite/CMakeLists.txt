paddle_test(
  pattern_rewrite_test
  SRCS
  pattern_rewrite_test.cc
  DEPS
  op_dialect_vjp)

paddle_test(drr_test SRCS drr_test.cc DEPS drr pir_transforms)

paddle_test(
  drr_same_type_binding_test
  SRCS
  drr_same_type_binding_test.cc
  DEPS
  drr
  op_dialect_vjp)

paddle_test(
  drr_fuse_linear_test
  SRCS
  drr_fuse_linear_test.cc
  DEPS
  pir_transforms
  drr)

paddle_test(
  drr_fuse_linear_param_grad_add_test
  SRCS
  drr_fuse_linear_param_grad_add_test.cc
  DEPS
  pir_transforms
  drr
  pir)

paddle_test(
  drr_attention_fuse_test
  SRCS
  drr_attention_fuse_test.cc
  DEPS
  pir_transforms
  drr
  pir)

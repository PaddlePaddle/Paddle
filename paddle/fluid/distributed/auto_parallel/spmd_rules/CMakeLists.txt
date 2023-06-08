cc_library(
  spmd_rule
  SRCS dist_tensor_spec.cc common.cc matmul_spmd_rule.cc
  DEPS dist_attr)

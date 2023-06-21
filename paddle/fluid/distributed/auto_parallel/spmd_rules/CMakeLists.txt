cc_library(
  spmd_rule
  SRCS common.cc dist_tensor_spec.cc matmul_spmd_rule.cc replicated_spmd_rule.cc
  DEPS op_dist_attr)

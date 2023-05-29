cc_library(
  dist_tensor_spec
  SRCS dist_tensor_spec.cc
  DEPS dist_attr)

cc_library(
  spmd_rule_base
  SRCS common.cc
  DEPS dist_tensor_spec)

cc_library(
  matmul_spmd_rule
  SRCS matmul_spmd_rule.cc
  DEPS spmd_rule_base)

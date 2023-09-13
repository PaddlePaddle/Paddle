cinn_cc_library(
  op_with_group_merge_pass
  SRCS
  group_with_group_merge_pass.cc
  op_with_group_merge_pass.cc
  tensor_node.cc
  DEPS
  cinn_op_dialect)

# cinn_cc_library(
#   group_with_group_merge_pass
#   SRCS group_with_group_merge_pass.cc
#   DEPS cinn_op_dialect)

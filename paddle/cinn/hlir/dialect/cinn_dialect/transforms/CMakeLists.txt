cc_library(
  op_with_group_merge_pass
  SRCS op_with_group_merge_pass.cc
  DEPS pd_dialect)

cc_library(
  group_with_group_merge_pass
  SRCS group_with_group_merge_pass.cc
  DEPS pd_dialect)

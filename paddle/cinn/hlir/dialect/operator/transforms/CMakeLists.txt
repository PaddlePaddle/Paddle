if(NOT CINN_ONLY)
  cinn_cc_library(
    op_with_group_merge_pass
    SRCS
    group_with_group_merge_pass.cc
    op_with_group_merge_pass.cc
    tensor_node.cc
    DEPS
    pd_op_dialect)
endif()

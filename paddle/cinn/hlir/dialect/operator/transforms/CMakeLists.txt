if(NOT CINN_ONLY)
  cinn_cc_library(
    op_with_group_merge_pass
    SRCS
    group_with_group_merge_pass.cc
    op_with_group_merge_pass.cc
    cinn_group_lowering_pass.cc
    tensor_node.cc
    DEPS
    pd_op_dialect
    pir_compiler
    cinn_runtime_dialect)
endif()

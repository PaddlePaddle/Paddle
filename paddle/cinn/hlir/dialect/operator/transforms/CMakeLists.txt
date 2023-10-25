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

  cinn_cc_library(
    pd_to_cinn_pass
    SRCS
    pd_to_cinn_pass.cc
    DEPS
    drr
    cinn_op_dialect
    pd_op_dialect)
endif()

if(NOT CINN_ONLY)
  cinn_cc_library(
    op_with_group_merge_pass
    SRCS
    group_with_group_merge_pass.cc
    op_with_group_merge_pass.cc
    cinn_group_lowering_pass.cc
    generate_shape_ops_to_run_first_pass.cc
    generate_shape_util.cc
    tensor_node.cc
    DEPS
    op_dialect_vjp
    pir_compiler
    cinn_runtime_dialect)
endif()

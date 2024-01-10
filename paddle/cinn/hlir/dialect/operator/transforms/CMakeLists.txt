add_subdirectory(group_merge)

if(NOT CINN_ONLY)

  cinn_cc_library(
    pd_to_cinn_pass
    SRCS
    pd_to_cinn_pass.cc
    DEPS
    drr
    cinn_op_dialect
    op_dialect_vjp)

  cinn_cc_library(
    add_broadcast_to_elementwise_pass
    SRCS
    add_broadcast_to_elementwise_pass.cc
    DEPS
    pir
    cinn_op_dialect
    op_dialect_vjp)

  cinn_cc_library(
    fuse_shape_ops_into_generate_shape_op_pass
    SRCS
    fuse_shape_ops_into_generate_shape_op_pass.cc
    DEPS
    pir
    cinn_op_dialect
    op_dialect_vjp)

  cinn_cc_library(
    fully_insert_broadcast_pass
    SRCS
    fully_insert_broadcast_pass.cc
    DEPS
    pir
    cinn_op_dialect
    op_dialect_vjp)

  cinn_cc_library(
    split_generate_shape_into_shape_ops_pass
    SRCS
    split_generate_shape_into_shape_ops_pass.cc
    DEPS
    pir
    cinn_op_dialect
    op_dialect_vjp)

endif()

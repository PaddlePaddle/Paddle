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
endif()

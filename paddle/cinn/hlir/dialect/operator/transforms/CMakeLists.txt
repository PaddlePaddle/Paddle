if(NOT CINN_ONLY)

  file(GLOB_RECURSE cinn_transforms_srcs "*.cc")

  set(cinn_transforms_deps
      pir
      drr
      op_dialect
      cinn_op_dialect
      op_dialect_vjp
      cinn_runtime_dialect
      pir_compiler)

  cc_library(
    cinn_transforms
    SRCS ${cinn_transforms_srcs}
    DEPS ${cinn_transforms_deps})

  cinn_cc_library(
    split_generate_shape_into_shape_ops_pass
    SRCS
    split_generate_shape_into_shape_ops_pass.cc
    DEPS
    pir
    cinn_op_dialect
    op_dialect_vjp)

endif()

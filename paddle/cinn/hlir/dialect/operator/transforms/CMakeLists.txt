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

  cinn_cc_library(cinn_transforms SRCS ${cinn_transforms_srcs} DEPS
                  ${cinn_transforms_deps})

endif()

set(CINN_PASS_ENTRY_DEPS op_dialect pir)

if(WITH_CINN)
  set(CINN_PASS_ENTRY_DEPS ${CINN_PASS_ENTRY_DEPS} cinn_op_dialect cinnapi
                           pir_transforms cinn_transforms sub_graph_checker)
endif()

cc_library(
  cinn_pass_entry
  SRCS cinn_pass_entry.cc
  DEPS ${CINN_PASS_ENTRY_DEPS})

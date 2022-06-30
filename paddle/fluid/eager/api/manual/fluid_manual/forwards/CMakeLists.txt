cc_library(
  fused_gate_attention_fwd_func
  SRCS fused_gate_attention_fwd_func.cc
  DEPS ${eager_deps} ${fluid_deps} ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS})

add_dependencies(fused_gate_attention_fwd_func eager_codegen)

set(fluid_manual_functions
    fused_gate_attention_fwd_func
    PARENT_SCOPE)

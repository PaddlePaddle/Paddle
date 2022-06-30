cc_library(
  fused_gate_attention_node
  SRCS fused_gate_attention_node.cc
  DEPS ${eager_deps} ${fluid_deps})

set(fluid_manual_nodes
    fused_gate_attention_node
    PARENT_SCOPE)

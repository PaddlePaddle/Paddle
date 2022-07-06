cc_library(
  fused_gate_attention_node
  SRCS fused_gate_attention_node.cc
  DEPS ${eager_deps} ${fluid_deps})

cc_library(
  fused_feedforward_node
  SRCS fused_feedforward_node.cc
  DEPS ${eager_deps} ${fluid_deps})

set(fluid_manual_nodes
    fused_gate_attention_node fused_feedforward_node
    PARENT_SCOPE)

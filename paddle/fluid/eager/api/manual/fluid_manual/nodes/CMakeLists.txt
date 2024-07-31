set(fluid_manual_nodes
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/fluid_manual/nodes/fused_gate_attention_node.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/fluid_manual/nodes/fused_feedforward_node.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/fluid_manual/nodes/fused_attention_node.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/fluid_manual/nodes/fused_gemm_epilogue_node.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/fluid_manual/nodes/fused_bias_dropout_residual_layer_norm_node.cc
    PARENT_SCOPE)

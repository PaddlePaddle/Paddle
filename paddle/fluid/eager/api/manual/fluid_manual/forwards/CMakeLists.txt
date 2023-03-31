set(fluid_manual_functions
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/fluid_manual/forwards/fused_gate_attention_fwd_func.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/fluid_manual/forwards/fused_feedforward_fwd_func.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/fluid_manual/forwards/fused_attention_fwd_func.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/fluid_manual/forwards/fused_gemm_epilogue_fwd_func.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/manual/fluid_manual/forwards/fused_bias_dropout_residual_layer_norm_fwd_func.cc
    PARENT_SCOPE)

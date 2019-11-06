include(operators)
register_operators(EXCLUDES
    conv_fusion_op
    fusion_transpose_flatten_concat_op
    fusion_conv_inception_op
    fused_fc_elementwise_layernorm_op
    multihead_matmul_op)

if (WITH_GPU)
    # conv_fusion_op needs cudnn 7 above
    if (NOT ${CUDNN_VERSION} VERSION_LESS 7100)
        op_library(conv_fusion_op)
        file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(conv2d_fusion);\n")
    endif()
    # fusion_transpose_flatten_concat_op
    op_library(fusion_transpose_flatten_concat_op)
    file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(fusion_transpose_flatten_concat);\n")
    # fusion_conv_inception_op needs cudnn 7 above
    if (NOT ${CUDNN_VERSION} VERSION_LESS 7100)
        op_library(fusion_conv_inception_op)
        file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(conv2d_inception_fusion);\n")
    endif()
    # fused_fc_elementwise_layernorm_op
    op_library(fused_fc_elementwise_layernorm_op)
    file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(fused_fc_elementwise_layernorm);\n")
    # multihead_matmul_op
    op_library(multihead_matmul_op)
    file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(multihead_matmul);\n")
endif()

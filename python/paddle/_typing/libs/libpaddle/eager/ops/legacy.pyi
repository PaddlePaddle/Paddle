from __future__ import annotations
__all__ = ['abs', 'abs_', 'accuracy', 'accuracy_check', 'acos', 'acos_', 'acosh', 'acosh_', 'adadelta', 'adagrad', 'adam', 'adamax', 'adamw', 'add_act_xpu', 'add_group_norm_silu', 'add_layernorm_xpu', 'add_position_encoding', 'addcmul_xpu', 'addmm', 'addmm_', 'affine_channel', 'affine_channel_', 'affine_grid', 'all_gather', 'all_reduce', 'all_to_all', 'allclose', 'anchor_generator', 'angle', 'apply_per_channel_scale', 'arg_max', 'arg_min', 'argsort', 'as_complex', 'as_real', 'as_strided', 'asgd', 'asin', 'asin_', 'asinh', 'asinh_', 'assert', 'assign', 'assign_pos', 'assign_value', 'atan', 'atan2', 'atan_', 'atanh', 'atanh_', 'attention_lstm', 'auc', 'average_accumulates', 'batch_fc', 'batch_norm', 'bce_loss', 'bce_loss_', 'beam_search', 'beam_search_decode', 'bernoulli', 'bicubic_interp_v2', 'bilinear_interp', 'bilinear_interp_v2', 'bilinear_tensor_product', 'bincount', 'binomial', 'bipartite_match', 'bitwise_and', 'bitwise_and_', 'bitwise_left_shift', 'bitwise_left_shift_', 'bitwise_not', 'bitwise_not_', 'bitwise_or', 'bitwise_or_', 'bitwise_right_shift', 'bitwise_right_shift_', 'bitwise_xor', 'bitwise_xor_', 'blha_get_max_len', 'block_multihead_attention', 'block_multihead_attention_xpu', 'bmm', 'bn_act_xpu', 'box_clip', 'box_coder', 'brelu', 'brelu_', 'broadcast', 'broadcast_tensors', 'c_comm_init_all', 'c_scatter', 'c_sync_calc_stream', 'c_sync_calc_stream_', 'calc_reduced_attn_scores', 'cast', 'ceil', 'ceil_', 'celu', 'channel_shuffle', 'check_finite_and_unscale', 'check_finite_and_unscale_', 'check_numerics', 'cholesky', 'cholesky_solve', 'chunk_eval', 'class_center_sample', 'clip', 'clip_', 'clip_by_norm', 'coalesce_tensor', 'collect_fpn_proposals', 'complex', 'concat', 'conj', 'conv1d_xpu', 'conv2d', 'conv2d_transpose', 'conv2d_transpose_bias', 'conv2d_transpose_xpu', 'conv2d_xpu', 'conv3d', 'conv3d_transpose', 'copysign', 'copysign_', 'correlation', 'cos', 'cos_', 'cosh', 'cosh_', 'crf_decoding', 'crop', 'crop_tensor', 'cross', 'cross_attention_xpu', 'cross_entropy', 'cross_entropy2', 'ctc_align', 'cudnn_lstm', 'cummax', 'cummin', 'cumprod', 'cumprod_', 'cumsum', 'cumsum_', 'cvm', 'data', 'decayed_adagrad', 'decode_jpeg', 'deformable_conv', 'deformable_conv_v1', 'depend', 'depthwise_conv2d', 'depthwise_conv2d_transpose', 'dequantize', 'dequantize_abs_max', 'dequantize_linear', 'dequantize_log', 'dequantize_xpu', 'detection_map', 'determinant', 'dgc', 'dgc_clip_by_norm', 'dgc_momentum', 'diag_embed', 'diag_v2', 'diagonal', 'digamma', 'digamma_', 'dirichlet', 'dist', 'dist_concat', 'distribute_fpn_proposals', 'distributed_fused_lamb_init', 'dot', 'dpsgd', 'dropout', 'edit_distance', 'eig', 'eigh', 'eigvals', 'eigvalsh', 'einsum', 'elementwise_add', 'elementwise_add_', 'elementwise_div', 'elementwise_floordiv', 'elementwise_fmax', 'elementwise_fmin', 'elementwise_heaviside', 'elementwise_max', 'elementwise_min', 'elementwise_mod', 'elementwise_mod_', 'elementwise_mul', 'elementwise_pow', 'elementwise_sub', 'elementwise_sub_', 'elu', 'elu_', 'embedding_with_eltwise_add_xpu', 'empty', 'equal', 'equal_all', 'erf', 'erf_', 'erfinv', 'erfinv_', 'exp', 'exp_', 'expand', 'expand_as_v2', 'expand_v2', 'expm1', 'expm1_', 'exponential', 'exponential_', 'eye', 'fake_channel_wise_dequantize_max_abs', 'fake_channel_wise_quantize_abs_max', 'fake_channel_wise_quantize_dequantize_abs_max', 'fake_dequantize_max_abs', 'fake_quantize_abs_max', 'fake_quantize_dequantize_abs_max', 'fake_quantize_dequantize_moving_average_abs_max', 'fake_quantize_dequantize_moving_average_abs_max_', 'fake_quantize_moving_average_abs_max', 'fake_quantize_moving_average_abs_max_', 'fake_quantize_range_abs_max', 'fake_quantize_range_abs_max_', 'fast_layernorm_xpu', 'fast_where_xpu', 'faster_tokenizer', 'fc', 'fc_xpu', 'feed', 'fetch', 'fetch_barrier', 'fetch_v2', 'fft_c2c', 'fft_c2r', 'fft_r2c', 'fill', 'fill_any', 'fill_any_', 'fill_any_like', 'fill_constant', 'fill_constant_batch_size_like', 'fill_diagonal', 'fill_diagonal_', 'fill_diagonal_tensor', 'fill_diagonal_tensor_', 'flash_attn', 'flash_attn_qkvpacked', 'flash_attn_unpadded', 'flash_attn_varlen_qkvpacked', 'flashmask_attention', 'flatten2', 'flatten2_', 'flatten_contiguous_range', 'flatten_contiguous_range_', 'flip', 'floor', 'floor_', 'fold', 'fp8_fp8_half_gemm_fused', 'fractional_max_pool2d', 'fractional_max_pool3d', 'frame', 'frobenius_norm', 'ftrl', 'full_int_array', 'fused_adam', 'fused_attention', 'fused_batch_norm_act', 'fused_bias_act', 'fused_bias_dropout_residual_layer_norm', 'fused_bias_residual_layernorm', 'fused_bn_add_activation', 'fused_conv2d', 'fused_conv2d_add_act', 'fused_conv3d', 'fused_dconv_drelu_dbn', 'fused_dot_product_attention', 'fused_dropout_add', 'fused_elementwise_add', 'fused_elementwise_div', 'fused_elementwise_mul', 'fused_elementwise_sub', 'fused_elemwise_activation', 'fused_elemwise_add_activation', 'fused_embedding_eltwise_layernorm', 'fused_embedding_fc_lstm', 'fused_fc_elementwise_layernorm', 'fused_feedforward', 'fused_gate_attention', 'fused_gemm_epilogue', 'fused_gemm_epilogue_grad', 'fused_linear_param_grad_add', 'fused_matmul', 'fused_moe', 'fused_multi_transformer', 'fused_multi_transformer_int8', 'fused_multi_transformer_int8_xpu', 'fused_multi_transformer_xpu', 'fused_rotary_position_embedding', 'fused_scale_bias_add_relu', 'fused_scale_bias_relu_conv_bn', 'fused_seqpool_cvm', 'fused_softmax_mask', 'fused_softmax_mask_upper_triangle', 'fused_token_prune', 'fused_transpose', 'fusion_group', 'fusion_gru', 'fusion_lstm', 'fusion_repeated_fc_relu', 'fusion_seqconv_eltadd_relu', 'fusion_seqexpand_concat_fc', 'fusion_seqpool_concat', 'fusion_seqpool_cvm_concat', 'fusion_squared_mat_sub', 'fusion_transpose_flatten_concat', 'gammaincc', 'gammaincc_', 'gammaln', 'gammaln_', 'gather', 'gather_nd', 'gather_tree', 'gaussian_inplace', 'gaussian_inplace_', 'gaussian_random', 'gelu', 'gemm_epilogue', 'generate_proposals', 'generate_proposals_v2', 'generate_sequence_xpu', 'get_core_ops_args_info', 'get_core_ops_args_type_info', 'get_core_ops_returns_info', 'get_tensor_from_selected_rows', 'grad_add', 'graph_khop_sampler', 'graph_reindex', 'graph_sample_neighbors', 'graph_send_recv', 'graph_send_ue_recv', 'graph_send_uv', 'greater_equal', 'greater_than', 'grid_sampler', 'group_norm', 'group_norm_silu_xpu', 'gru', 'gru_unit', 'gumbel_softmax', 'hard_shrink', 'hard_sigmoid', 'hard_swish', 'hash', 'hierarchical_sigmoid', 'hinge_loss', 'histogram', 'huber_loss', 'i0', 'i0_', 'i0e', 'i1', 'i1e', 'identity_loss', 'identity_loss_', 'im2sequence', 'imag', 'increment', 'index_add', 'index_add_', 'index_put', 'index_put_', 'index_sample', 'index_select', 'index_select_strided', 'instance_norm', 'inverse', 'is_empty', 'isclose', 'isfinite_v2', 'isinf_v2', 'isnan_v2', 'kldiv_loss', 'kron', 'kthvalue', 'l1_norm', 'l1_norm_', 'label_smooth', 'lamb', 'lars_momentum', 'layer_norm', 'layer_norm_act_xpu', 'leaky_relu', 'leaky_relu_', 'lerp', 'lerp_', 'less_equal', 'less_than', 'lgamma', 'lgamma_', 'limit_by_capacity', 'linear_interp_v2', 'linspace', 'llm_int8_linear', 'load', 'load_combine', 'lod_reset', 'lod_reset_', 'log', 'log10', 'log10_', 'log1p', 'log1p_', 'log2', 'log2_', 'log_', 'log_loss', 'log_softmax', 'logcumsumexp', 'logical_and', 'logical_and_', 'logical_not', 'logical_not_', 'logical_or', 'logical_or_', 'logical_xor', 'logical_xor_', 'logit', 'logit_', 'logsigmoid', 'logspace', 'logsumexp', 'lookup_table', 'lookup_table_dequant', 'lookup_table_v2', 'lp_pool2d', 'lrn', 'lstm', 'lstsq', 'lu', 'lu_', 'lu_unpack', 'margin_cross_entropy', 'mask_adaptive_xpu', 'masked_multihead_attention', 'masked_select', 'match_matrix_tensor', 'matmul', 'matmul_v2', 'matrix_nms', 'matrix_power', 'matrix_rank', 'matrix_rank_atol_rtol', 'max_pool2d_v2', 'max_pool2d_with_index', 'max_pool3d_with_index', 'maxout', 'mean', 'memcpy', 'memcpy_d2h', 'memcpy_h2d', 'memory_efficient_attention', 'merge_selected_rows', 'merged_adam', 'merged_momentum', 'meshgrid', 'mish', 'mode', 'momentum', 'moving_average_abs_max_scale', 'mul', 'multi_dot', 'multi_encoder_xpu', 'multi_gru', 'multiclass_nms', 'multiclass_nms3', 'multihead_matmul', 'multinomial', 'multiplex', 'mv', 'nadam', 'nanmedian', 'nce', 'nearest_interp', 'nearest_interp_v2', 'nextafter', 'nll_loss', 'nms', 'nop', 'norm', 'not_equal', 'npu_identity', 'number_count', 'one_hot_v2', 'overlap_add', 'p_norm', 'p_recv', 'p_recv_array', 'p_send', 'p_send_array', 'pad', 'pad2d_xpu', 'pad3d', 'partial_concat', 'partial_sum', 'pir_run_program', 'pixel_shuffle', 'pixel_unshuffle', 'poisson', 'polygamma', 'polygamma_', 'pool2d', 'pool3d', 'pow', 'pow_', 'prelu', 'prior_box', 'prune_gate_by_capacity', 'psroi_pool', 'pull_box_sparse', 'pull_gpups_sparse', 'pull_sparse_v2', 'push_dense', 'put_along_axis', 'put_along_axis_', 'pyramid_hash', 'qkv_attention_xpu', 'qkv_unpack_mha', 'qr', 'quant_linear', 'quantize', 'quantize_linear', 'quantize_xpu', 'radam', 'randint', 'random_routing', 'random_routing_', 'randperm', 'range', 'rank_attention', 'read_file', 'real', 'reciprocal', 'reciprocal_', 'reduce', 'reduce_all', 'reduce_amax', 'reduce_amin', 'reduce_any', 'reduce_as', 'reduce_max', 'reduce_mean', 'reduce_min', 'reduce_prod', 'reduce_scatter', 'reduce_sum', 'relu', 'relu6', 'relu_', 'renorm', 'renorm_', 'repeat_interleave', 'requantize', 'reshape', 'reshape2', 'reshape2_', 'reshape_', 'resnet_basic_block', 'resnet_unit', 'reverse', 'rms_norm', 'rmsprop', 'rnn', 'roformer_relative_embedding_xpu', 'roi_align', 'roi_pool', 'roll', 'round', 'round_', 'row_conv', 'rprop', 'rrelu', 'rsqrt', 'rsqrt_', 'run_program', 'save', 'save_combine', 'scale', 'scale_', 'scatter', 'scatter_', 'scatter_nd_add', 'searchsorted', 'seed', 'segment_pool', 'self_dp_attention', 'selu', 'sequence_conv', 'sequence_expand', 'sequence_mask', 'sequence_pool', 'sequence_softmax', 'sequence_unpad_xpu', 'set_value', 'set_value_', 'sgd', 'shadow_output', 'shape', 'shard_index', 'share_buffer', 'share_data', 'shuffle_batch', 'shuffle_channel', 'sigmoid', 'sigmoid_', 'sigmoid_cross_entropy_with_logits', 'sigmoid_cross_entropy_with_logits_', 'sign', 'silu', 'sin', 'sin_', 'sine_pos_xpu', 'sinh', 'sinh_', 'size', 'skip_layernorm', 'slice', 'slogdeterminant', 'soft_relu', 'soft_relu_', 'softmax', 'softmax_', 'softmax_with_cross_entropy', 'softmax_with_cross_entropy_', 'softplus', 'softshrink', 'softsign', 'solve', 'sparse_attention', 'sparse_momentum', 'spatial_transformer_resblock_xpu', 'spectral_norm', 'split', 'sqrt', 'sqrt_', 'square', 'squared_l2_norm', 'squeeze2', 'squeeze2_', 'squeeze_excitation_block', 'stack', 'standard_gamma', 'stanh', 'stft', 'strided_slice', 'sum', 'svd', 'swiglu', 'swish', 'sync_batch_norm', 'take_along_axis', 'tan', 'tan_', 'tanh', 'tanh_', 'tanh_shrink', 'tdm_child', 'tdm_sampler', 'temporal_shift', 'tensor_unfold', 'thresholded_relu', 'thresholded_relu_', 'tile', 'top_k', 'top_k_v2', 'top_p_sampling', 'trace', 'transfer_layout', 'transpose', 'transpose2', 'triangular_solve', 'tril_indices', 'tril_triu', 'trilinear_interp_v2', 'triu_indices', 'trunc', 'trunc_', 'truncated_gaussian_random', 'unbind', 'unfold', 'uniform_random', 'uniform_random_batch_size_like', 'uniform_random_inplace', 'uniform_random_inplace_', 'unique', 'unique_consecutive', 'unpool', 'unpool3d', 'unsqueeze2', 'unsqueeze2_', 'unstack', 'update_loss_scaling', 'variable_length_memory_efficient_attention', 'view_dtype', 'view_shape', 'viterbi_decode', 'warpctc', 'warprnnt', 'weight_dequantize', 'weight_only_linear', 'weight_only_linear_xpu', 'weight_quantize', 'weighted_sample_neighbors', 'where', 'where_', 'where_index', 'yolo_box', 'yolo_box_head', 'yolo_box_post', 'yolo_box_xpu', 'yolov3_loss']
def abs(*args, **kwargs):
    """
    C++ interface function for abs in dygraph.
    """
def abs_(*args, **kwargs):
    """
    C++ interface function for abs_ in dygraph.
    """
def accuracy(*args, **kwargs):
    """
    C++ interface function for accuracy in dygraph.
    """
def accuracy_check(*args, **kwargs):
    """
    C++ interface function for accuracy_check in dygraph.
    """
def acos(*args, **kwargs):
    """
    C++ interface function for acos in dygraph.
    """
def acos_(*args, **kwargs):
    """
    C++ interface function for acos_ in dygraph.
    """
def acosh(*args, **kwargs):
    """
    C++ interface function for acosh in dygraph.
    """
def acosh_(*args, **kwargs):
    """
    C++ interface function for acosh_ in dygraph.
    """
def adadelta(*args, **kwargs):
    """
    C++ interface function for adadelta in dygraph.
    """
def adagrad(*args, **kwargs):
    """
    C++ interface function for adagrad in dygraph.
    """
def adam(*args, **kwargs):
    """
    C++ interface function for adam in dygraph.
    """
def adamax(*args, **kwargs):
    """
    C++ interface function for adamax in dygraph.
    """
def adamw(*args, **kwargs):
    """
    C++ interface function for adamw in dygraph.
    """
def add_act_xpu(*args, **kwargs):
    """
    C++ interface function for add_act_xpu in dygraph.
    """
def add_group_norm_silu(*args, **kwargs):
    """
    C++ interface function for add_group_norm_silu in dygraph.
    """
def add_layernorm_xpu(*args, **kwargs):
    """
    C++ interface function for add_layernorm_xpu in dygraph.
    """
def add_position_encoding(*args, **kwargs):
    """
    C++ interface function for add_position_encoding in dygraph.
    """
def addcmul_xpu(*args, **kwargs):
    """
    C++ interface function for addcmul_xpu in dygraph.
    """
def addmm(*args, **kwargs):
    """
    C++ interface function for addmm in dygraph.
    """
def addmm_(*args, **kwargs):
    """
    C++ interface function for addmm_ in dygraph.
    """
def affine_channel(*args, **kwargs):
    """
    C++ interface function for affine_channel in dygraph.
    """
def affine_channel_(*args, **kwargs):
    """
    C++ interface function for affine_channel_ in dygraph.
    """
def affine_grid(*args, **kwargs):
    """
    C++ interface function for affine_grid in dygraph.
    """
def all_gather(*args, **kwargs):
    """
    C++ interface function for all_gather in dygraph.
    """
def all_reduce(*args, **kwargs):
    """
    C++ interface function for all_reduce in dygraph.
    """
def all_to_all(*args, **kwargs):
    """
    C++ interface function for all_to_all in dygraph.
    """
def allclose(*args, **kwargs):
    """
    C++ interface function for allclose in dygraph.
    """
def anchor_generator(*args, **kwargs):
    """
    C++ interface function for anchor_generator in dygraph.
    """
def angle(*args, **kwargs):
    """
    C++ interface function for angle in dygraph.
    """
def apply_per_channel_scale(*args, **kwargs):
    """
    C++ interface function for apply_per_channel_scale in dygraph.
    """
def arg_max(*args, **kwargs):
    """
    C++ interface function for arg_max in dygraph.
    """
def arg_min(*args, **kwargs):
    """
    C++ interface function for arg_min in dygraph.
    """
def argsort(*args, **kwargs):
    """
    C++ interface function for argsort in dygraph.
    """
def as_complex(*args, **kwargs):
    """
    C++ interface function for as_complex in dygraph.
    """
def as_real(*args, **kwargs):
    """
    C++ interface function for as_real in dygraph.
    """
def as_strided(*args, **kwargs):
    """
    C++ interface function for as_strided in dygraph.
    """
def asgd(*args, **kwargs):
    """
    C++ interface function for asgd in dygraph.
    """
def asin(*args, **kwargs):
    """
    C++ interface function for asin in dygraph.
    """
def asin_(*args, **kwargs):
    """
    C++ interface function for asin_ in dygraph.
    """
def asinh(*args, **kwargs):
    """
    C++ interface function for asinh in dygraph.
    """
def asinh_(*args, **kwargs):
    """
    C++ interface function for asinh_ in dygraph.
    """
def assert_(*args, **kwargs):
    """
    C++ interface function for assert in dygraph.
    """
def assign(*args, **kwargs):
    """
    C++ interface function for assign in dygraph.
    """
def assign_pos(*args, **kwargs):
    """
    C++ interface function for assign_pos in dygraph.
    """
def assign_value(*args, **kwargs):
    """
    C++ interface function for assign_value in dygraph.
    """
def atan(*args, **kwargs):
    """
    C++ interface function for atan in dygraph.
    """
def atan2(*args, **kwargs):
    """
    C++ interface function for atan2 in dygraph.
    """
def atan_(*args, **kwargs):
    """
    C++ interface function for atan_ in dygraph.
    """
def atanh(*args, **kwargs):
    """
    C++ interface function for atanh in dygraph.
    """
def atanh_(*args, **kwargs):
    """
    C++ interface function for atanh_ in dygraph.
    """
def attention_lstm(*args, **kwargs):
    """
    C++ interface function for attention_lstm in dygraph.
    """
def auc(*args, **kwargs):
    """
    C++ interface function for auc in dygraph.
    """
def average_accumulates(*args, **kwargs):
    """
    C++ interface function for average_accumulates in dygraph.
    """
def batch_fc(*args, **kwargs):
    """
    C++ interface function for batch_fc in dygraph.
    """
def batch_norm(*args, **kwargs):
    """
    C++ interface function for batch_norm in dygraph.
    """
def bce_loss(*args, **kwargs):
    """
    C++ interface function for bce_loss in dygraph.
    """
def bce_loss_(*args, **kwargs):
    """
    C++ interface function for bce_loss_ in dygraph.
    """
def beam_search(*args, **kwargs):
    """
    C++ interface function for beam_search in dygraph.
    """
def beam_search_decode(*args, **kwargs):
    """
    C++ interface function for beam_search_decode in dygraph.
    """
def bernoulli(*args, **kwargs):
    """
    C++ interface function for bernoulli in dygraph.
    """
def bicubic_interp_v2(*args, **kwargs):
    """
    C++ interface function for bicubic_interp_v2 in dygraph.
    """
def bilinear_interp(*args, **kwargs):
    """
    C++ interface function for bilinear_interp in dygraph.
    """
def bilinear_interp_v2(*args, **kwargs):
    """
    C++ interface function for bilinear_interp_v2 in dygraph.
    """
def bilinear_tensor_product(*args, **kwargs):
    """
    C++ interface function for bilinear_tensor_product in dygraph.
    """
def bincount(*args, **kwargs):
    """
    C++ interface function for bincount in dygraph.
    """
def binomial(*args, **kwargs):
    """
    C++ interface function for binomial in dygraph.
    """
def bipartite_match(*args, **kwargs):
    """
    C++ interface function for bipartite_match in dygraph.
    """
def bitwise_and(*args, **kwargs):
    """
    C++ interface function for bitwise_and in dygraph.
    """
def bitwise_and_(*args, **kwargs):
    """
    C++ interface function for bitwise_and_ in dygraph.
    """
def bitwise_left_shift(*args, **kwargs):
    """
    C++ interface function for bitwise_left_shift in dygraph.
    """
def bitwise_left_shift_(*args, **kwargs):
    """
    C++ interface function for bitwise_left_shift_ in dygraph.
    """
def bitwise_not(*args, **kwargs):
    """
    C++ interface function for bitwise_not in dygraph.
    """
def bitwise_not_(*args, **kwargs):
    """
    C++ interface function for bitwise_not_ in dygraph.
    """
def bitwise_or(*args, **kwargs):
    """
    C++ interface function for bitwise_or in dygraph.
    """
def bitwise_or_(*args, **kwargs):
    """
    C++ interface function for bitwise_or_ in dygraph.
    """
def bitwise_right_shift(*args, **kwargs):
    """
    C++ interface function for bitwise_right_shift in dygraph.
    """
def bitwise_right_shift_(*args, **kwargs):
    """
    C++ interface function for bitwise_right_shift_ in dygraph.
    """
def bitwise_xor(*args, **kwargs):
    """
    C++ interface function for bitwise_xor in dygraph.
    """
def bitwise_xor_(*args, **kwargs):
    """
    C++ interface function for bitwise_xor_ in dygraph.
    """
def blha_get_max_len(*args, **kwargs):
    """
    C++ interface function for blha_get_max_len in dygraph.
    """
def block_multihead_attention(*args, **kwargs):
    """
    C++ interface function for block_multihead_attention in dygraph.
    """
def block_multihead_attention_xpu(*args, **kwargs):
    """
    C++ interface function for block_multihead_attention_xpu in dygraph.
    """
def bmm(*args, **kwargs):
    """
    C++ interface function for bmm in dygraph.
    """
def bn_act_xpu(*args, **kwargs):
    """
    C++ interface function for bn_act_xpu in dygraph.
    """
def box_clip(*args, **kwargs):
    """
    C++ interface function for box_clip in dygraph.
    """
def box_coder(*args, **kwargs):
    """
    C++ interface function for box_coder in dygraph.
    """
def brelu(*args, **kwargs):
    """
    C++ interface function for brelu in dygraph.
    """
def brelu_(*args, **kwargs):
    """
    C++ interface function for brelu_ in dygraph.
    """
def broadcast(*args, **kwargs):
    """
    C++ interface function for broadcast in dygraph.
    """
def broadcast_tensors(*args, **kwargs):
    """
    C++ interface function for broadcast_tensors in dygraph.
    """
def c_comm_init_all(*args, **kwargs):
    """
    C++ interface function for c_comm_init_all in dygraph.
    """
def c_scatter(*args, **kwargs):
    """
    C++ interface function for c_scatter in dygraph.
    """
def c_sync_calc_stream(*args, **kwargs):
    """
    C++ interface function for c_sync_calc_stream in dygraph.
    """
def c_sync_calc_stream_(*args, **kwargs):
    """
    C++ interface function for c_sync_calc_stream_ in dygraph.
    """
def calc_reduced_attn_scores(*args, **kwargs):
    """
    C++ interface function for calc_reduced_attn_scores in dygraph.
    """
def cast(*args, **kwargs):
    """
    C++ interface function for cast in dygraph.
    """
def ceil(*args, **kwargs):
    """
    C++ interface function for ceil in dygraph.
    """
def ceil_(*args, **kwargs):
    """
    C++ interface function for ceil_ in dygraph.
    """
def celu(*args, **kwargs):
    """
    C++ interface function for celu in dygraph.
    """
def channel_shuffle(*args, **kwargs):
    """
    C++ interface function for channel_shuffle in dygraph.
    """
def check_finite_and_unscale(*args, **kwargs):
    """
    C++ interface function for check_finite_and_unscale in dygraph.
    """
def check_finite_and_unscale_(*args, **kwargs):
    """
    C++ interface function for check_finite_and_unscale_ in dygraph.
    """
def check_numerics(*args, **kwargs):
    """
    C++ interface function for check_numerics in dygraph.
    """
def cholesky(*args, **kwargs):
    """
    C++ interface function for cholesky in dygraph.
    """
def cholesky_solve(*args, **kwargs):
    """
    C++ interface function for cholesky_solve in dygraph.
    """
def chunk_eval(*args, **kwargs):
    """
    C++ interface function for chunk_eval in dygraph.
    """
def class_center_sample(*args, **kwargs):
    """
    C++ interface function for class_center_sample in dygraph.
    """
def clip(*args, **kwargs):
    """
    C++ interface function for clip in dygraph.
    """
def clip_(*args, **kwargs):
    """
    C++ interface function for clip_ in dygraph.
    """
def clip_by_norm(*args, **kwargs):
    """
    C++ interface function for clip_by_norm in dygraph.
    """
def coalesce_tensor(*args, **kwargs):
    """
    C++ interface function for coalesce_tensor in dygraph.
    """
def collect_fpn_proposals(*args, **kwargs):
    """
    C++ interface function for collect_fpn_proposals in dygraph.
    """
def complex(*args, **kwargs):
    """
    C++ interface function for complex in dygraph.
    """
def concat(*args, **kwargs):
    """
    C++ interface function for concat in dygraph.
    """
def conj(*args, **kwargs):
    """
    C++ interface function for conj in dygraph.
    """
def conv1d_xpu(*args, **kwargs):
    """
    C++ interface function for conv1d_xpu in dygraph.
    """
def conv2d(*args, **kwargs):
    """
    C++ interface function for conv2d in dygraph.
    """
def conv2d_transpose(*args, **kwargs):
    """
    C++ interface function for conv2d_transpose in dygraph.
    """
def conv2d_transpose_bias(*args, **kwargs):
    """
    C++ interface function for conv2d_transpose_bias in dygraph.
    """
def conv2d_transpose_xpu(*args, **kwargs):
    """
    C++ interface function for conv2d_transpose_xpu in dygraph.
    """
def conv2d_xpu(*args, **kwargs):
    """
    C++ interface function for conv2d_xpu in dygraph.
    """
def conv3d(*args, **kwargs):
    """
    C++ interface function for conv3d in dygraph.
    """
def conv3d_transpose(*args, **kwargs):
    """
    C++ interface function for conv3d_transpose in dygraph.
    """
def copysign(*args, **kwargs):
    """
    C++ interface function for copysign in dygraph.
    """
def copysign_(*args, **kwargs):
    """
    C++ interface function for copysign_ in dygraph.
    """
def correlation(*args, **kwargs):
    """
    C++ interface function for correlation in dygraph.
    """
def cos(*args, **kwargs):
    """
    C++ interface function for cos in dygraph.
    """
def cos_(*args, **kwargs):
    """
    C++ interface function for cos_ in dygraph.
    """
def cosh(*args, **kwargs):
    """
    C++ interface function for cosh in dygraph.
    """
def cosh_(*args, **kwargs):
    """
    C++ interface function for cosh_ in dygraph.
    """
def crf_decoding(*args, **kwargs):
    """
    C++ interface function for crf_decoding in dygraph.
    """
def crop(*args, **kwargs):
    """
    C++ interface function for crop in dygraph.
    """
def crop_tensor(*args, **kwargs):
    """
    C++ interface function for crop_tensor in dygraph.
    """
def cross(*args, **kwargs):
    """
    C++ interface function for cross in dygraph.
    """
def cross_attention_xpu(*args, **kwargs):
    """
    C++ interface function for cross_attention_xpu in dygraph.
    """
def cross_entropy(*args, **kwargs):
    """
    C++ interface function for cross_entropy in dygraph.
    """
def cross_entropy2(*args, **kwargs):
    """
    C++ interface function for cross_entropy2 in dygraph.
    """
def ctc_align(*args, **kwargs):
    """
    C++ interface function for ctc_align in dygraph.
    """
def cudnn_lstm(*args, **kwargs):
    """
    C++ interface function for cudnn_lstm in dygraph.
    """
def cummax(*args, **kwargs):
    """
    C++ interface function for cummax in dygraph.
    """
def cummin(*args, **kwargs):
    """
    C++ interface function for cummin in dygraph.
    """
def cumprod(*args, **kwargs):
    """
    C++ interface function for cumprod in dygraph.
    """
def cumprod_(*args, **kwargs):
    """
    C++ interface function for cumprod_ in dygraph.
    """
def cumsum(*args, **kwargs):
    """
    C++ interface function for cumsum in dygraph.
    """
def cumsum_(*args, **kwargs):
    """
    C++ interface function for cumsum_ in dygraph.
    """
def cvm(*args, **kwargs):
    """
    C++ interface function for cvm in dygraph.
    """
def data(*args, **kwargs):
    """
    C++ interface function for data in dygraph.
    """
def decayed_adagrad(*args, **kwargs):
    """
    C++ interface function for decayed_adagrad in dygraph.
    """
def decode_jpeg(*args, **kwargs):
    """
    C++ interface function for decode_jpeg in dygraph.
    """
def deformable_conv(*args, **kwargs):
    """
    C++ interface function for deformable_conv in dygraph.
    """
def deformable_conv_v1(*args, **kwargs):
    """
    C++ interface function for deformable_conv_v1 in dygraph.
    """
def depend(*args, **kwargs):
    """
    C++ interface function for depend in dygraph.
    """
def depthwise_conv2d(*args, **kwargs):
    """
    C++ interface function for depthwise_conv2d in dygraph.
    """
def depthwise_conv2d_transpose(*args, **kwargs):
    """
    C++ interface function for depthwise_conv2d_transpose in dygraph.
    """
def dequantize(*args, **kwargs):
    """
    C++ interface function for dequantize in dygraph.
    """
def dequantize_abs_max(*args, **kwargs):
    """
    C++ interface function for dequantize_abs_max in dygraph.
    """
def dequantize_linear(*args, **kwargs):
    """
    C++ interface function for dequantize_linear in dygraph.
    """
def dequantize_log(*args, **kwargs):
    """
    C++ interface function for dequantize_log in dygraph.
    """
def dequantize_xpu(*args, **kwargs):
    """
    C++ interface function for dequantize_xpu in dygraph.
    """
def detection_map(*args, **kwargs):
    """
    C++ interface function for detection_map in dygraph.
    """
def determinant(*args, **kwargs):
    """
    C++ interface function for determinant in dygraph.
    """
def dgc(*args, **kwargs):
    """
    C++ interface function for dgc in dygraph.
    """
def dgc_clip_by_norm(*args, **kwargs):
    """
    C++ interface function for dgc_clip_by_norm in dygraph.
    """
def dgc_momentum(*args, **kwargs):
    """
    C++ interface function for dgc_momentum in dygraph.
    """
def diag_embed(*args, **kwargs):
    """
    C++ interface function for diag_embed in dygraph.
    """
def diag_v2(*args, **kwargs):
    """
    C++ interface function for diag_v2 in dygraph.
    """
def diagonal(*args, **kwargs):
    """
    C++ interface function for diagonal in dygraph.
    """
def digamma(*args, **kwargs):
    """
    C++ interface function for digamma in dygraph.
    """
def digamma_(*args, **kwargs):
    """
    C++ interface function for digamma_ in dygraph.
    """
def dirichlet(*args, **kwargs):
    """
    C++ interface function for dirichlet in dygraph.
    """
def dist(*args, **kwargs):
    """
    C++ interface function for dist in dygraph.
    """
def dist_concat(*args, **kwargs):
    """
    C++ interface function for dist_concat in dygraph.
    """
def distribute_fpn_proposals(*args, **kwargs):
    """
    C++ interface function for distribute_fpn_proposals in dygraph.
    """
def distributed_fused_lamb_init(*args, **kwargs):
    """
    C++ interface function for distributed_fused_lamb_init in dygraph.
    """
def dot(*args, **kwargs):
    """
    C++ interface function for dot in dygraph.
    """
def dpsgd(*args, **kwargs):
    """
    C++ interface function for dpsgd in dygraph.
    """
def dropout(*args, **kwargs):
    """
    C++ interface function for dropout in dygraph.
    """
def edit_distance(*args, **kwargs):
    """
    C++ interface function for edit_distance in dygraph.
    """
def eig(*args, **kwargs):
    """
    C++ interface function for eig in dygraph.
    """
def eigh(*args, **kwargs):
    """
    C++ interface function for eigh in dygraph.
    """
def eigvals(*args, **kwargs):
    """
    C++ interface function for eigvals in dygraph.
    """
def eigvalsh(*args, **kwargs):
    """
    C++ interface function for eigvalsh in dygraph.
    """
def einsum(*args, **kwargs):
    """
    C++ interface function for einsum in dygraph.
    """
def elementwise_add(*args, **kwargs):
    """
    C++ interface function for elementwise_add in dygraph.
    """
def elementwise_add_(*args, **kwargs):
    """
    C++ interface function for elementwise_add_ in dygraph.
    """
def elementwise_div(*args, **kwargs):
    """
    C++ interface function for elementwise_div in dygraph.
    """
def elementwise_floordiv(*args, **kwargs):
    """
    C++ interface function for elementwise_floordiv in dygraph.
    """
def elementwise_fmax(*args, **kwargs):
    """
    C++ interface function for elementwise_fmax in dygraph.
    """
def elementwise_fmin(*args, **kwargs):
    """
    C++ interface function for elementwise_fmin in dygraph.
    """
def elementwise_heaviside(*args, **kwargs):
    """
    C++ interface function for elementwise_heaviside in dygraph.
    """
def elementwise_max(*args, **kwargs):
    """
    C++ interface function for elementwise_max in dygraph.
    """
def elementwise_min(*args, **kwargs):
    """
    C++ interface function for elementwise_min in dygraph.
    """
def elementwise_mod(*args, **kwargs):
    """
    C++ interface function for elementwise_mod in dygraph.
    """
def elementwise_mod_(*args, **kwargs):
    """
    C++ interface function for elementwise_mod_ in dygraph.
    """
def elementwise_mul(*args, **kwargs):
    """
    C++ interface function for elementwise_mul in dygraph.
    """
def elementwise_pow(*args, **kwargs):
    """
    C++ interface function for elementwise_pow in dygraph.
    """
def elementwise_sub(*args, **kwargs):
    """
    C++ interface function for elementwise_sub in dygraph.
    """
def elementwise_sub_(*args, **kwargs):
    """
    C++ interface function for elementwise_sub_ in dygraph.
    """
def elu(*args, **kwargs):
    """
    C++ interface function for elu in dygraph.
    """
def elu_(*args, **kwargs):
    """
    C++ interface function for elu_ in dygraph.
    """
def embedding_with_eltwise_add_xpu(*args, **kwargs):
    """
    C++ interface function for embedding_with_eltwise_add_xpu in dygraph.
    """
def empty(*args, **kwargs):
    """
    C++ interface function for empty in dygraph.
    """
def equal(*args, **kwargs):
    """
    C++ interface function for equal in dygraph.
    """
def equal_all(*args, **kwargs):
    """
    C++ interface function for equal_all in dygraph.
    """
def erf(*args, **kwargs):
    """
    C++ interface function for erf in dygraph.
    """
def erf_(*args, **kwargs):
    """
    C++ interface function for erf_ in dygraph.
    """
def erfinv(*args, **kwargs):
    """
    C++ interface function for erfinv in dygraph.
    """
def erfinv_(*args, **kwargs):
    """
    C++ interface function for erfinv_ in dygraph.
    """
def exp(*args, **kwargs):
    """
    C++ interface function for exp in dygraph.
    """
def exp_(*args, **kwargs):
    """
    C++ interface function for exp_ in dygraph.
    """
def expand(*args, **kwargs):
    """
    C++ interface function for expand in dygraph.
    """
def expand_as_v2(*args, **kwargs):
    """
    C++ interface function for expand_as_v2 in dygraph.
    """
def expand_v2(*args, **kwargs):
    """
    C++ interface function for expand_v2 in dygraph.
    """
def expm1(*args, **kwargs):
    """
    C++ interface function for expm1 in dygraph.
    """
def expm1_(*args, **kwargs):
    """
    C++ interface function for expm1_ in dygraph.
    """
def exponential(*args, **kwargs):
    """
    C++ interface function for exponential in dygraph.
    """
def exponential_(*args, **kwargs):
    """
    C++ interface function for exponential_ in dygraph.
    """
def eye(*args, **kwargs):
    """
    C++ interface function for eye in dygraph.
    """
def fake_channel_wise_dequantize_max_abs(*args, **kwargs):
    """
    C++ interface function for fake_channel_wise_dequantize_max_abs in dygraph.
    """
def fake_channel_wise_quantize_abs_max(*args, **kwargs):
    """
    C++ interface function for fake_channel_wise_quantize_abs_max in dygraph.
    """
def fake_channel_wise_quantize_dequantize_abs_max(*args, **kwargs):
    """
    C++ interface function for fake_channel_wise_quantize_dequantize_abs_max in dygraph.
    """
def fake_dequantize_max_abs(*args, **kwargs):
    """
    C++ interface function for fake_dequantize_max_abs in dygraph.
    """
def fake_quantize_abs_max(*args, **kwargs):
    """
    C++ interface function for fake_quantize_abs_max in dygraph.
    """
def fake_quantize_dequantize_abs_max(*args, **kwargs):
    """
    C++ interface function for fake_quantize_dequantize_abs_max in dygraph.
    """
def fake_quantize_dequantize_moving_average_abs_max(*args, **kwargs):
    """
    C++ interface function for fake_quantize_dequantize_moving_average_abs_max in dygraph.
    """
def fake_quantize_dequantize_moving_average_abs_max_(*args, **kwargs):
    """
    C++ interface function for fake_quantize_dequantize_moving_average_abs_max_ in dygraph.
    """
def fake_quantize_moving_average_abs_max(*args, **kwargs):
    """
    C++ interface function for fake_quantize_moving_average_abs_max in dygraph.
    """
def fake_quantize_moving_average_abs_max_(*args, **kwargs):
    """
    C++ interface function for fake_quantize_moving_average_abs_max_ in dygraph.
    """
def fake_quantize_range_abs_max(*args, **kwargs):
    """
    C++ interface function for fake_quantize_range_abs_max in dygraph.
    """
def fake_quantize_range_abs_max_(*args, **kwargs):
    """
    C++ interface function for fake_quantize_range_abs_max_ in dygraph.
    """
def fast_layernorm_xpu(*args, **kwargs):
    """
    C++ interface function for fast_layernorm_xpu in dygraph.
    """
def fast_where_xpu(*args, **kwargs):
    """
    C++ interface function for fast_where_xpu in dygraph.
    """
def faster_tokenizer(*args, **kwargs):
    """
    C++ interface function for faster_tokenizer in dygraph.
    """
def fc(*args, **kwargs):
    """
    C++ interface function for fc in dygraph.
    """
def fc_xpu(*args, **kwargs):
    """
    C++ interface function for fc_xpu in dygraph.
    """
def feed(*args, **kwargs):
    """
    C++ interface function for feed in dygraph.
    """
def fetch(*args, **kwargs):
    """
    C++ interface function for fetch in dygraph.
    """
def fetch_barrier(*args, **kwargs):
    """
    C++ interface function for fetch_barrier in dygraph.
    """
def fetch_v2(*args, **kwargs):
    """
    C++ interface function for fetch_v2 in dygraph.
    """
def fft_c2c(*args, **kwargs):
    """
    C++ interface function for fft_c2c in dygraph.
    """
def fft_c2r(*args, **kwargs):
    """
    C++ interface function for fft_c2r in dygraph.
    """
def fft_r2c(*args, **kwargs):
    """
    C++ interface function for fft_r2c in dygraph.
    """
def fill(*args, **kwargs):
    """
    C++ interface function for fill in dygraph.
    """
def fill_any(*args, **kwargs):
    """
    C++ interface function for fill_any in dygraph.
    """
def fill_any_(*args, **kwargs):
    """
    C++ interface function for fill_any_ in dygraph.
    """
def fill_any_like(*args, **kwargs):
    """
    C++ interface function for fill_any_like in dygraph.
    """
def fill_constant(*args, **kwargs):
    """
    C++ interface function for fill_constant in dygraph.
    """
def fill_constant_batch_size_like(*args, **kwargs):
    """
    C++ interface function for fill_constant_batch_size_like in dygraph.
    """
def fill_diagonal(*args, **kwargs):
    """
    C++ interface function for fill_diagonal in dygraph.
    """
def fill_diagonal_(*args, **kwargs):
    """
    C++ interface function for fill_diagonal_ in dygraph.
    """
def fill_diagonal_tensor(*args, **kwargs):
    """
    C++ interface function for fill_diagonal_tensor in dygraph.
    """
def fill_diagonal_tensor_(*args, **kwargs):
    """
    C++ interface function for fill_diagonal_tensor_ in dygraph.
    """
def flash_attn(*args, **kwargs):
    """
    C++ interface function for flash_attn in dygraph.
    """
def flash_attn_qkvpacked(*args, **kwargs):
    """
    C++ interface function for flash_attn_qkvpacked in dygraph.
    """
def flash_attn_unpadded(*args, **kwargs):
    """
    C++ interface function for flash_attn_unpadded in dygraph.
    """
def flash_attn_varlen_qkvpacked(*args, **kwargs):
    """
    C++ interface function for flash_attn_varlen_qkvpacked in dygraph.
    """
def flashmask_attention(*args, **kwargs):
    """
    C++ interface function for flashmask_attention in dygraph.
    """
def flatten2(*args, **kwargs):
    """
    C++ interface function for flatten2 in dygraph.
    """
def flatten2_(*args, **kwargs):
    """
    C++ interface function for flatten2_ in dygraph.
    """
def flatten_contiguous_range(*args, **kwargs):
    """
    C++ interface function for flatten_contiguous_range in dygraph.
    """
def flatten_contiguous_range_(*args, **kwargs):
    """
    C++ interface function for flatten_contiguous_range_ in dygraph.
    """
def flip(*args, **kwargs):
    """
    C++ interface function for flip in dygraph.
    """
def floor(*args, **kwargs):
    """
    C++ interface function for floor in dygraph.
    """
def floor_(*args, **kwargs):
    """
    C++ interface function for floor_ in dygraph.
    """
def fold(*args, **kwargs):
    """
    C++ interface function for fold in dygraph.
    """
def fp8_fp8_half_gemm_fused(*args, **kwargs):
    """
    C++ interface function for fp8_fp8_half_gemm_fused in dygraph.
    """
def fractional_max_pool2d(*args, **kwargs):
    """
    C++ interface function for fractional_max_pool2d in dygraph.
    """
def fractional_max_pool3d(*args, **kwargs):
    """
    C++ interface function for fractional_max_pool3d in dygraph.
    """
def frame(*args, **kwargs):
    """
    C++ interface function for frame in dygraph.
    """
def frobenius_norm(*args, **kwargs):
    """
    C++ interface function for frobenius_norm in dygraph.
    """
def ftrl(*args, **kwargs):
    """
    C++ interface function for ftrl in dygraph.
    """
def full_int_array(*args, **kwargs):
    """
    C++ interface function for full_int_array in dygraph.
    """
def fused_adam(*args, **kwargs):
    """
    C++ interface function for fused_adam in dygraph.
    """
def fused_attention(*args, **kwargs):
    """
    C++ interface function for fused_attention in dygraph.
    """
def fused_batch_norm_act(*args, **kwargs):
    """
    C++ interface function for fused_batch_norm_act in dygraph.
    """
def fused_bias_act(*args, **kwargs):
    """
    C++ interface function for fused_bias_act in dygraph.
    """
def fused_bias_dropout_residual_layer_norm(*args, **kwargs):
    """
    C++ interface function for fused_bias_dropout_residual_layer_norm in dygraph.
    """
def fused_bias_residual_layernorm(*args, **kwargs):
    """
    C++ interface function for fused_bias_residual_layernorm in dygraph.
    """
def fused_bn_add_activation(*args, **kwargs):
    """
    C++ interface function for fused_bn_add_activation in dygraph.
    """
def fused_conv2d(*args, **kwargs):
    """
    C++ interface function for fused_conv2d in dygraph.
    """
def fused_conv2d_add_act(*args, **kwargs):
    """
    C++ interface function for fused_conv2d_add_act in dygraph.
    """
def fused_conv3d(*args, **kwargs):
    """
    C++ interface function for fused_conv3d in dygraph.
    """
def fused_dconv_drelu_dbn(*args, **kwargs):
    """
    C++ interface function for fused_dconv_drelu_dbn in dygraph.
    """
def fused_dot_product_attention(*args, **kwargs):
    """
    C++ interface function for fused_dot_product_attention in dygraph.
    """
def fused_dropout_add(*args, **kwargs):
    """
    C++ interface function for fused_dropout_add in dygraph.
    """
def fused_elementwise_add(*args, **kwargs):
    """
    C++ interface function for fused_elementwise_add in dygraph.
    """
def fused_elementwise_div(*args, **kwargs):
    """
    C++ interface function for fused_elementwise_div in dygraph.
    """
def fused_elementwise_mul(*args, **kwargs):
    """
    C++ interface function for fused_elementwise_mul in dygraph.
    """
def fused_elementwise_sub(*args, **kwargs):
    """
    C++ interface function for fused_elementwise_sub in dygraph.
    """
def fused_elemwise_activation(*args, **kwargs):
    """
    C++ interface function for fused_elemwise_activation in dygraph.
    """
def fused_elemwise_add_activation(*args, **kwargs):
    """
    C++ interface function for fused_elemwise_add_activation in dygraph.
    """
def fused_embedding_eltwise_layernorm(*args, **kwargs):
    """
    C++ interface function for fused_embedding_eltwise_layernorm in dygraph.
    """
def fused_embedding_fc_lstm(*args, **kwargs):
    """
    C++ interface function for fused_embedding_fc_lstm in dygraph.
    """
def fused_fc_elementwise_layernorm(*args, **kwargs):
    """
    C++ interface function for fused_fc_elementwise_layernorm in dygraph.
    """
def fused_feedforward(*args, **kwargs):
    """
    C++ interface function for fused_feedforward in dygraph.
    """
def fused_gate_attention(*args, **kwargs):
    """
    C++ interface function for fused_gate_attention in dygraph.
    """
def fused_gemm_epilogue(*args, **kwargs):
    """
    C++ interface function for fused_gemm_epilogue in dygraph.
    """
def fused_gemm_epilogue_grad(*args, **kwargs):
    """
    C++ interface function for fused_gemm_epilogue_grad in dygraph.
    """
def fused_linear_param_grad_add(*args, **kwargs):
    """
    C++ interface function for fused_linear_param_grad_add in dygraph.
    """
def fused_matmul(*args, **kwargs):
    """
    C++ interface function for fused_matmul in dygraph.
    """
def fused_moe(*args, **kwargs):
    """
    C++ interface function for fused_moe in dygraph.
    """
def fused_multi_transformer(*args, **kwargs):
    """
    C++ interface function for fused_multi_transformer in dygraph.
    """
def fused_multi_transformer_int8(*args, **kwargs):
    """
    C++ interface function for fused_multi_transformer_int8 in dygraph.
    """
def fused_multi_transformer_int8_xpu(*args, **kwargs):
    """
    C++ interface function for fused_multi_transformer_int8_xpu in dygraph.
    """
def fused_multi_transformer_xpu(*args, **kwargs):
    """
    C++ interface function for fused_multi_transformer_xpu in dygraph.
    """
def fused_rotary_position_embedding(*args, **kwargs):
    """
    C++ interface function for fused_rotary_position_embedding in dygraph.
    """
def fused_scale_bias_add_relu(*args, **kwargs):
    """
    C++ interface function for fused_scale_bias_add_relu in dygraph.
    """
def fused_scale_bias_relu_conv_bn(*args, **kwargs):
    """
    C++ interface function for fused_scale_bias_relu_conv_bn in dygraph.
    """
def fused_seqpool_cvm(*args, **kwargs):
    """
    C++ interface function for fused_seqpool_cvm in dygraph.
    """
def fused_softmax_mask(*args, **kwargs):
    """
    C++ interface function for fused_softmax_mask in dygraph.
    """
def fused_softmax_mask_upper_triangle(*args, **kwargs):
    """
    C++ interface function for fused_softmax_mask_upper_triangle in dygraph.
    """
def fused_token_prune(*args, **kwargs):
    """
    C++ interface function for fused_token_prune in dygraph.
    """
def fused_transpose(*args, **kwargs):
    """
    C++ interface function for fused_transpose in dygraph.
    """
def fusion_group(*args, **kwargs):
    """
    C++ interface function for fusion_group in dygraph.
    """
def fusion_gru(*args, **kwargs):
    """
    C++ interface function for fusion_gru in dygraph.
    """
def fusion_lstm(*args, **kwargs):
    """
    C++ interface function for fusion_lstm in dygraph.
    """
def fusion_repeated_fc_relu(*args, **kwargs):
    """
    C++ interface function for fusion_repeated_fc_relu in dygraph.
    """
def fusion_seqconv_eltadd_relu(*args, **kwargs):
    """
    C++ interface function for fusion_seqconv_eltadd_relu in dygraph.
    """
def fusion_seqexpand_concat_fc(*args, **kwargs):
    """
    C++ interface function for fusion_seqexpand_concat_fc in dygraph.
    """
def fusion_seqpool_concat(*args, **kwargs):
    """
    C++ interface function for fusion_seqpool_concat in dygraph.
    """
def fusion_seqpool_cvm_concat(*args, **kwargs):
    """
    C++ interface function for fusion_seqpool_cvm_concat in dygraph.
    """
def fusion_squared_mat_sub(*args, **kwargs):
    """
    C++ interface function for fusion_squared_mat_sub in dygraph.
    """
def fusion_transpose_flatten_concat(*args, **kwargs):
    """
    C++ interface function for fusion_transpose_flatten_concat in dygraph.
    """
def gammaincc(*args, **kwargs):
    """
    C++ interface function for gammaincc in dygraph.
    """
def gammaincc_(*args, **kwargs):
    """
    C++ interface function for gammaincc_ in dygraph.
    """
def gammaln(*args, **kwargs):
    """
    C++ interface function for gammaln in dygraph.
    """
def gammaln_(*args, **kwargs):
    """
    C++ interface function for gammaln_ in dygraph.
    """
def gather(*args, **kwargs):
    """
    C++ interface function for gather in dygraph.
    """
def gather_nd(*args, **kwargs):
    """
    C++ interface function for gather_nd in dygraph.
    """
def gather_tree(*args, **kwargs):
    """
    C++ interface function for gather_tree in dygraph.
    """
def gaussian_inplace(*args, **kwargs):
    """
    C++ interface function for gaussian_inplace in dygraph.
    """
def gaussian_inplace_(*args, **kwargs):
    """
    C++ interface function for gaussian_inplace_ in dygraph.
    """
def gaussian_random(*args, **kwargs):
    """
    C++ interface function for gaussian_random in dygraph.
    """
def gelu(*args, **kwargs):
    """
    C++ interface function for gelu in dygraph.
    """
def gemm_epilogue(*args, **kwargs):
    """
    C++ interface function for gemm_epilogue in dygraph.
    """
def generate_proposals(*args, **kwargs):
    """
    C++ interface function for generate_proposals in dygraph.
    """
def generate_proposals_v2(*args, **kwargs):
    """
    C++ interface function for generate_proposals_v2 in dygraph.
    """
def generate_sequence_xpu(*args, **kwargs):
    """
    C++ interface function for generate_sequence_xpu in dygraph.
    """
def get_core_ops_args_info(*args, **kwargs):
    """
    C++ interface function for eager_get_core_ops_args_info.
    """
def get_core_ops_args_type_info(*args, **kwargs):
    """
    C++ interface function for eager_get_core_ops_args_type_info.
    """
def get_core_ops_returns_info(*args, **kwargs):
    """
    C++ interface function for eager_get_core_ops_returns_info.
    """
def get_tensor_from_selected_rows(*args, **kwargs):
    """
    C++ interface function for get_tensor_from_selected_rows in dygraph.
    """
def grad_add(*args, **kwargs):
    """
    C++ interface function for grad_add in dygraph.
    """
def graph_khop_sampler(*args, **kwargs):
    """
    C++ interface function for graph_khop_sampler in dygraph.
    """
def graph_reindex(*args, **kwargs):
    """
    C++ interface function for graph_reindex in dygraph.
    """
def graph_sample_neighbors(*args, **kwargs):
    """
    C++ interface function for graph_sample_neighbors in dygraph.
    """
def graph_send_recv(*args, **kwargs):
    """
    C++ interface function for graph_send_recv in dygraph.
    """
def graph_send_ue_recv(*args, **kwargs):
    """
    C++ interface function for graph_send_ue_recv in dygraph.
    """
def graph_send_uv(*args, **kwargs):
    """
    C++ interface function for graph_send_uv in dygraph.
    """
def greater_equal(*args, **kwargs):
    """
    C++ interface function for greater_equal in dygraph.
    """
def greater_than(*args, **kwargs):
    """
    C++ interface function for greater_than in dygraph.
    """
def grid_sampler(*args, **kwargs):
    """
    C++ interface function for grid_sampler in dygraph.
    """
def group_norm(*args, **kwargs):
    """
    C++ interface function for group_norm in dygraph.
    """
def group_norm_silu_xpu(*args, **kwargs):
    """
    C++ interface function for group_norm_silu_xpu in dygraph.
    """
def gru(*args, **kwargs):
    """
    C++ interface function for gru in dygraph.
    """
def gru_unit(*args, **kwargs):
    """
    C++ interface function for gru_unit in dygraph.
    """
def gumbel_softmax(*args, **kwargs):
    """
    C++ interface function for gumbel_softmax in dygraph.
    """
def hard_shrink(*args, **kwargs):
    """
    C++ interface function for hard_shrink in dygraph.
    """
def hard_sigmoid(*args, **kwargs):
    """
    C++ interface function for hard_sigmoid in dygraph.
    """
def hard_swish(*args, **kwargs):
    """
    C++ interface function for hard_swish in dygraph.
    """
def hash(*args, **kwargs):
    """
    C++ interface function for hash in dygraph.
    """
def hierarchical_sigmoid(*args, **kwargs):
    """
    C++ interface function for hierarchical_sigmoid in dygraph.
    """
def hinge_loss(*args, **kwargs):
    """
    C++ interface function for hinge_loss in dygraph.
    """
def histogram(*args, **kwargs):
    """
    C++ interface function for histogram in dygraph.
    """
def huber_loss(*args, **kwargs):
    """
    C++ interface function for huber_loss in dygraph.
    """
def i0(*args, **kwargs):
    """
    C++ interface function for i0 in dygraph.
    """
def i0_(*args, **kwargs):
    """
    C++ interface function for i0_ in dygraph.
    """
def i0e(*args, **kwargs):
    """
    C++ interface function for i0e in dygraph.
    """
def i1(*args, **kwargs):
    """
    C++ interface function for i1 in dygraph.
    """
def i1e(*args, **kwargs):
    """
    C++ interface function for i1e in dygraph.
    """
def identity_loss(*args, **kwargs):
    """
    C++ interface function for identity_loss in dygraph.
    """
def identity_loss_(*args, **kwargs):
    """
    C++ interface function for identity_loss_ in dygraph.
    """
def im2sequence(*args, **kwargs):
    """
    C++ interface function for im2sequence in dygraph.
    """
def imag(*args, **kwargs):
    """
    C++ interface function for imag in dygraph.
    """
def increment(*args, **kwargs):
    """
    C++ interface function for increment in dygraph.
    """
def index_add(*args, **kwargs):
    """
    C++ interface function for index_add in dygraph.
    """
def index_add_(*args, **kwargs):
    """
    C++ interface function for index_add_ in dygraph.
    """
def index_put(*args, **kwargs):
    """
    C++ interface function for index_put in dygraph.
    """
def index_put_(*args, **kwargs):
    """
    C++ interface function for index_put_ in dygraph.
    """
def index_sample(*args, **kwargs):
    """
    C++ interface function for index_sample in dygraph.
    """
def index_select(*args, **kwargs):
    """
    C++ interface function for index_select in dygraph.
    """
def index_select_strided(*args, **kwargs):
    """
    C++ interface function for index_select_strided in dygraph.
    """
def instance_norm(*args, **kwargs):
    """
    C++ interface function for instance_norm in dygraph.
    """
def inverse(*args, **kwargs):
    """
    C++ interface function for inverse in dygraph.
    """
def is_empty(*args, **kwargs):
    """
    C++ interface function for is_empty in dygraph.
    """
def isclose(*args, **kwargs):
    """
    C++ interface function for isclose in dygraph.
    """
def isfinite_v2(*args, **kwargs):
    """
    C++ interface function for isfinite_v2 in dygraph.
    """
def isinf_v2(*args, **kwargs):
    """
    C++ interface function for isinf_v2 in dygraph.
    """
def isnan_v2(*args, **kwargs):
    """
    C++ interface function for isnan_v2 in dygraph.
    """
def kldiv_loss(*args, **kwargs):
    """
    C++ interface function for kldiv_loss in dygraph.
    """
def kron(*args, **kwargs):
    """
    C++ interface function for kron in dygraph.
    """
def kthvalue(*args, **kwargs):
    """
    C++ interface function for kthvalue in dygraph.
    """
def l1_norm(*args, **kwargs):
    """
    C++ interface function for l1_norm in dygraph.
    """
def l1_norm_(*args, **kwargs):
    """
    C++ interface function for l1_norm_ in dygraph.
    """
def label_smooth(*args, **kwargs):
    """
    C++ interface function for label_smooth in dygraph.
    """
def lamb(*args, **kwargs):
    """
    C++ interface function for lamb in dygraph.
    """
def lars_momentum(*args, **kwargs):
    """
    C++ interface function for lars_momentum in dygraph.
    """
def layer_norm(*args, **kwargs):
    """
    C++ interface function for layer_norm in dygraph.
    """
def layer_norm_act_xpu(*args, **kwargs):
    """
    C++ interface function for layer_norm_act_xpu in dygraph.
    """
def leaky_relu(*args, **kwargs):
    """
    C++ interface function for leaky_relu in dygraph.
    """
def leaky_relu_(*args, **kwargs):
    """
    C++ interface function for leaky_relu_ in dygraph.
    """
def lerp(*args, **kwargs):
    """
    C++ interface function for lerp in dygraph.
    """
def lerp_(*args, **kwargs):
    """
    C++ interface function for lerp_ in dygraph.
    """
def less_equal(*args, **kwargs):
    """
    C++ interface function for less_equal in dygraph.
    """
def less_than(*args, **kwargs):
    """
    C++ interface function for less_than in dygraph.
    """
def lgamma(*args, **kwargs):
    """
    C++ interface function for lgamma in dygraph.
    """
def lgamma_(*args, **kwargs):
    """
    C++ interface function for lgamma_ in dygraph.
    """
def limit_by_capacity(*args, **kwargs):
    """
    C++ interface function for limit_by_capacity in dygraph.
    """
def linear_interp_v2(*args, **kwargs):
    """
    C++ interface function for linear_interp_v2 in dygraph.
    """
def linspace(*args, **kwargs):
    """
    C++ interface function for linspace in dygraph.
    """
def llm_int8_linear(*args, **kwargs):
    """
    C++ interface function for llm_int8_linear in dygraph.
    """
def load(*args, **kwargs):
    """
    C++ interface function for load in dygraph.
    """
def load_combine(*args, **kwargs):
    """
    C++ interface function for load_combine in dygraph.
    """
def lod_reset(*args, **kwargs):
    """
    C++ interface function for lod_reset in dygraph.
    """
def lod_reset_(*args, **kwargs):
    """
    C++ interface function for lod_reset_ in dygraph.
    """
def log(*args, **kwargs):
    """
    C++ interface function for log in dygraph.
    """
def log10(*args, **kwargs):
    """
    C++ interface function for log10 in dygraph.
    """
def log10_(*args, **kwargs):
    """
    C++ interface function for log10_ in dygraph.
    """
def log1p(*args, **kwargs):
    """
    C++ interface function for log1p in dygraph.
    """
def log1p_(*args, **kwargs):
    """
    C++ interface function for log1p_ in dygraph.
    """
def log2(*args, **kwargs):
    """
    C++ interface function for log2 in dygraph.
    """
def log2_(*args, **kwargs):
    """
    C++ interface function for log2_ in dygraph.
    """
def log_(*args, **kwargs):
    """
    C++ interface function for log_ in dygraph.
    """
def log_loss(*args, **kwargs):
    """
    C++ interface function for log_loss in dygraph.
    """
def log_softmax(*args, **kwargs):
    """
    C++ interface function for log_softmax in dygraph.
    """
def logcumsumexp(*args, **kwargs):
    """
    C++ interface function for logcumsumexp in dygraph.
    """
def logical_and(*args, **kwargs):
    """
    C++ interface function for logical_and in dygraph.
    """
def logical_and_(*args, **kwargs):
    """
    C++ interface function for logical_and_ in dygraph.
    """
def logical_not(*args, **kwargs):
    """
    C++ interface function for logical_not in dygraph.
    """
def logical_not_(*args, **kwargs):
    """
    C++ interface function for logical_not_ in dygraph.
    """
def logical_or(*args, **kwargs):
    """
    C++ interface function for logical_or in dygraph.
    """
def logical_or_(*args, **kwargs):
    """
    C++ interface function for logical_or_ in dygraph.
    """
def logical_xor(*args, **kwargs):
    """
    C++ interface function for logical_xor in dygraph.
    """
def logical_xor_(*args, **kwargs):
    """
    C++ interface function for logical_xor_ in dygraph.
    """
def logit(*args, **kwargs):
    """
    C++ interface function for logit in dygraph.
    """
def logit_(*args, **kwargs):
    """
    C++ interface function for logit_ in dygraph.
    """
def logsigmoid(*args, **kwargs):
    """
    C++ interface function for logsigmoid in dygraph.
    """
def logspace(*args, **kwargs):
    """
    C++ interface function for logspace in dygraph.
    """
def logsumexp(*args, **kwargs):
    """
    C++ interface function for logsumexp in dygraph.
    """
def lookup_table(*args, **kwargs):
    """
    C++ interface function for lookup_table in dygraph.
    """
def lookup_table_dequant(*args, **kwargs):
    """
    C++ interface function for lookup_table_dequant in dygraph.
    """
def lookup_table_v2(*args, **kwargs):
    """
    C++ interface function for lookup_table_v2 in dygraph.
    """
def lp_pool2d(*args, **kwargs):
    """
    C++ interface function for lp_pool2d in dygraph.
    """
def lrn(*args, **kwargs):
    """
    C++ interface function for lrn in dygraph.
    """
def lstm(*args, **kwargs):
    """
    C++ interface function for lstm in dygraph.
    """
def lstsq(*args, **kwargs):
    """
    C++ interface function for lstsq in dygraph.
    """
def lu(*args, **kwargs):
    """
    C++ interface function for lu in dygraph.
    """
def lu_(*args, **kwargs):
    """
    C++ interface function for lu_ in dygraph.
    """
def lu_unpack(*args, **kwargs):
    """
    C++ interface function for lu_unpack in dygraph.
    """
def margin_cross_entropy(*args, **kwargs):
    """
    C++ interface function for margin_cross_entropy in dygraph.
    """
def mask_adaptive_xpu(*args, **kwargs):
    """
    C++ interface function for mask_adaptive_xpu in dygraph.
    """
def masked_multihead_attention(*args, **kwargs):
    """
    C++ interface function for masked_multihead_attention in dygraph.
    """
def masked_select(*args, **kwargs):
    """
    C++ interface function for masked_select in dygraph.
    """
def match_matrix_tensor(*args, **kwargs):
    """
    C++ interface function for match_matrix_tensor in dygraph.
    """
def matmul(*args, **kwargs):
    """
    C++ interface function for matmul in dygraph.
    """
def matmul_v2(*args, **kwargs):
    """
    C++ interface function for matmul_v2 in dygraph.
    """
def matrix_nms(*args, **kwargs):
    """
    C++ interface function for matrix_nms in dygraph.
    """
def matrix_power(*args, **kwargs):
    """
    C++ interface function for matrix_power in dygraph.
    """
def matrix_rank(*args, **kwargs):
    """
    C++ interface function for matrix_rank in dygraph.
    """
def matrix_rank_atol_rtol(*args, **kwargs):
    """
    C++ interface function for matrix_rank_atol_rtol in dygraph.
    """
def max_pool2d_v2(*args, **kwargs):
    """
    C++ interface function for max_pool2d_v2 in dygraph.
    """
def max_pool2d_with_index(*args, **kwargs):
    """
    C++ interface function for max_pool2d_with_index in dygraph.
    """
def max_pool3d_with_index(*args, **kwargs):
    """
    C++ interface function for max_pool3d_with_index in dygraph.
    """
def maxout(*args, **kwargs):
    """
    C++ interface function for maxout in dygraph.
    """
def mean(*args, **kwargs):
    """
    C++ interface function for mean in dygraph.
    """
def memcpy(*args, **kwargs):
    """
    C++ interface function for memcpy in dygraph.
    """
def memcpy_d2h(*args, **kwargs):
    """
    C++ interface function for memcpy_d2h in dygraph.
    """
def memcpy_h2d(*args, **kwargs):
    """
    C++ interface function for memcpy_h2d in dygraph.
    """
def memory_efficient_attention(*args, **kwargs):
    """
    C++ interface function for memory_efficient_attention in dygraph.
    """
def merge_selected_rows(*args, **kwargs):
    """
    C++ interface function for merge_selected_rows in dygraph.
    """
def merged_adam(*args, **kwargs):
    """
    C++ interface function for merged_adam in dygraph.
    """
def merged_momentum(*args, **kwargs):
    """
    C++ interface function for merged_momentum in dygraph.
    """
def meshgrid(*args, **kwargs):
    """
    C++ interface function for meshgrid in dygraph.
    """
def mish(*args, **kwargs):
    """
    C++ interface function for mish in dygraph.
    """
def mode(*args, **kwargs):
    """
    C++ interface function for mode in dygraph.
    """
def momentum(*args, **kwargs):
    """
    C++ interface function for momentum in dygraph.
    """
def moving_average_abs_max_scale(*args, **kwargs):
    """
    C++ interface function for moving_average_abs_max_scale in dygraph.
    """
def mul(*args, **kwargs):
    """
    C++ interface function for mul in dygraph.
    """
def multi_dot(*args, **kwargs):
    """
    C++ interface function for multi_dot in dygraph.
    """
def multi_encoder_xpu(*args, **kwargs):
    """
    C++ interface function for multi_encoder_xpu in dygraph.
    """
def multi_gru(*args, **kwargs):
    """
    C++ interface function for multi_gru in dygraph.
    """
def multiclass_nms(*args, **kwargs):
    """
    C++ interface function for multiclass_nms in dygraph.
    """
def multiclass_nms3(*args, **kwargs):
    """
    C++ interface function for multiclass_nms3 in dygraph.
    """
def multihead_matmul(*args, **kwargs):
    """
    C++ interface function for multihead_matmul in dygraph.
    """
def multinomial(*args, **kwargs):
    """
    C++ interface function for multinomial in dygraph.
    """
def multiplex(*args, **kwargs):
    """
    C++ interface function for multiplex in dygraph.
    """
def mv(*args, **kwargs):
    """
    C++ interface function for mv in dygraph.
    """
def nadam(*args, **kwargs):
    """
    C++ interface function for nadam in dygraph.
    """
def nanmedian(*args, **kwargs):
    """
    C++ interface function for nanmedian in dygraph.
    """
def nce(*args, **kwargs):
    """
    C++ interface function for nce in dygraph.
    """
def nearest_interp(*args, **kwargs):
    """
    C++ interface function for nearest_interp in dygraph.
    """
def nearest_interp_v2(*args, **kwargs):
    """
    C++ interface function for nearest_interp_v2 in dygraph.
    """
def nextafter(*args, **kwargs):
    """
    C++ interface function for nextafter in dygraph.
    """
def nll_loss(*args, **kwargs):
    """
    C++ interface function for nll_loss in dygraph.
    """
def nms(*args, **kwargs):
    """
    C++ interface function for nms in dygraph.
    """
def nop(*args, **kwargs):
    """
    C++ interface function for nop in dygraph.
    """
def norm(*args, **kwargs):
    """
    C++ interface function for norm in dygraph.
    """
def not_equal(*args, **kwargs):
    """
    C++ interface function for not_equal in dygraph.
    """
def npu_identity(*args, **kwargs):
    """
    C++ interface function for npu_identity in dygraph.
    """
def number_count(*args, **kwargs):
    """
    C++ interface function for number_count in dygraph.
    """
def one_hot_v2(*args, **kwargs):
    """
    C++ interface function for one_hot_v2 in dygraph.
    """
def overlap_add(*args, **kwargs):
    """
    C++ interface function for overlap_add in dygraph.
    """
def p_norm(*args, **kwargs):
    """
    C++ interface function for p_norm in dygraph.
    """
def p_recv(*args, **kwargs):
    """
    C++ interface function for p_recv in dygraph.
    """
def p_recv_array(*args, **kwargs):
    """
    C++ interface function for p_recv_array in dygraph.
    """
def p_send(*args, **kwargs):
    """
    C++ interface function for p_send in dygraph.
    """
def p_send_array(*args, **kwargs):
    """
    C++ interface function for p_send_array in dygraph.
    """
def pad(*args, **kwargs):
    """
    C++ interface function for pad in dygraph.
    """
def pad2d_xpu(*args, **kwargs):
    """
    C++ interface function for pad2d_xpu in dygraph.
    """
def pad3d(*args, **kwargs):
    """
    C++ interface function for pad3d in dygraph.
    """
def partial_concat(*args, **kwargs):
    """
    C++ interface function for partial_concat in dygraph.
    """
def partial_sum(*args, **kwargs):
    """
    C++ interface function for partial_sum in dygraph.
    """
def pir_run_program(*args, **kwargs):
    """
    C++ interface function for run_program in dygraph.
    """
def pixel_shuffle(*args, **kwargs):
    """
    C++ interface function for pixel_shuffle in dygraph.
    """
def pixel_unshuffle(*args, **kwargs):
    """
    C++ interface function for pixel_unshuffle in dygraph.
    """
def poisson(*args, **kwargs):
    """
    C++ interface function for poisson in dygraph.
    """
def polygamma(*args, **kwargs):
    """
    C++ interface function for polygamma in dygraph.
    """
def polygamma_(*args, **kwargs):
    """
    C++ interface function for polygamma_ in dygraph.
    """
def pool2d(*args, **kwargs):
    """
    C++ interface function for pool2d in dygraph.
    """
def pool3d(*args, **kwargs):
    """
    C++ interface function for pool3d in dygraph.
    """
def pow(*args, **kwargs):
    """
    C++ interface function for pow in dygraph.
    """
def pow_(*args, **kwargs):
    """
    C++ interface function for pow_ in dygraph.
    """
def prelu(*args, **kwargs):
    """
    C++ interface function for prelu in dygraph.
    """
def prior_box(*args, **kwargs):
    """
    C++ interface function for prior_box in dygraph.
    """
def prune_gate_by_capacity(*args, **kwargs):
    """
    C++ interface function for prune_gate_by_capacity in dygraph.
    """
def psroi_pool(*args, **kwargs):
    """
    C++ interface function for psroi_pool in dygraph.
    """
def pull_box_sparse(*args, **kwargs):
    """
    C++ interface function for pull_box_sparse in dygraph.
    """
def pull_gpups_sparse(*args, **kwargs):
    """
    C++ interface function for pull_gpups_sparse in dygraph.
    """
def pull_sparse_v2(*args, **kwargs):
    """
    C++ interface function for pull_sparse_v2 in dygraph.
    """
def push_dense(*args, **kwargs):
    """
    C++ interface function for push_dense in dygraph.
    """
def put_along_axis(*args, **kwargs):
    """
    C++ interface function for put_along_axis in dygraph.
    """
def put_along_axis_(*args, **kwargs):
    """
    C++ interface function for put_along_axis_ in dygraph.
    """
def pyramid_hash(*args, **kwargs):
    """
    C++ interface function for pyramid_hash in dygraph.
    """
def qkv_attention_xpu(*args, **kwargs):
    """
    C++ interface function for qkv_attention_xpu in dygraph.
    """
def qkv_unpack_mha(*args, **kwargs):
    """
    C++ interface function for qkv_unpack_mha in dygraph.
    """
def qr(*args, **kwargs):
    """
    C++ interface function for qr in dygraph.
    """
def quant_linear(*args, **kwargs):
    """
    C++ interface function for quant_linear in dygraph.
    """
def quantize(*args, **kwargs):
    """
    C++ interface function for quantize in dygraph.
    """
def quantize_linear(*args, **kwargs):
    """
    C++ interface function for quantize_linear in dygraph.
    """
def quantize_xpu(*args, **kwargs):
    """
    C++ interface function for quantize_xpu in dygraph.
    """
def radam(*args, **kwargs):
    """
    C++ interface function for radam in dygraph.
    """
def randint(*args, **kwargs):
    """
    C++ interface function for randint in dygraph.
    """
def random_routing(*args, **kwargs):
    """
    C++ interface function for random_routing in dygraph.
    """
def random_routing_(*args, **kwargs):
    """
    C++ interface function for random_routing_ in dygraph.
    """
def randperm(*args, **kwargs):
    """
    C++ interface function for randperm in dygraph.
    """
def range(*args, **kwargs):
    """
    C++ interface function for range in dygraph.
    """
def rank_attention(*args, **kwargs):
    """
    C++ interface function for rank_attention in dygraph.
    """
def read_file(*args, **kwargs):
    """
    C++ interface function for read_file in dygraph.
    """
def real(*args, **kwargs):
    """
    C++ interface function for real in dygraph.
    """
def reciprocal(*args, **kwargs):
    """
    C++ interface function for reciprocal in dygraph.
    """
def reciprocal_(*args, **kwargs):
    """
    C++ interface function for reciprocal_ in dygraph.
    """
def reduce(*args, **kwargs):
    """
    C++ interface function for reduce in dygraph.
    """
def reduce_all(*args, **kwargs):
    """
    C++ interface function for reduce_all in dygraph.
    """
def reduce_amax(*args, **kwargs):
    """
    C++ interface function for reduce_amax in dygraph.
    """
def reduce_amin(*args, **kwargs):
    """
    C++ interface function for reduce_amin in dygraph.
    """
def reduce_any(*args, **kwargs):
    """
    C++ interface function for reduce_any in dygraph.
    """
def reduce_as(*args, **kwargs):
    """
    C++ interface function for reduce_as in dygraph.
    """
def reduce_max(*args, **kwargs):
    """
    C++ interface function for reduce_max in dygraph.
    """
def reduce_mean(*args, **kwargs):
    """
    C++ interface function for reduce_mean in dygraph.
    """
def reduce_min(*args, **kwargs):
    """
    C++ interface function for reduce_min in dygraph.
    """
def reduce_prod(*args, **kwargs):
    """
    C++ interface function for reduce_prod in dygraph.
    """
def reduce_scatter(*args, **kwargs):
    """
    C++ interface function for reduce_scatter in dygraph.
    """
def reduce_sum(*args, **kwargs):
    """
    C++ interface function for reduce_sum in dygraph.
    """
def relu(*args, **kwargs):
    """
    C++ interface function for relu in dygraph.
    """
def relu6(*args, **kwargs):
    """
    C++ interface function for relu6 in dygraph.
    """
def relu_(*args, **kwargs):
    """
    C++ interface function for relu_ in dygraph.
    """
def renorm(*args, **kwargs):
    """
    C++ interface function for renorm in dygraph.
    """
def renorm_(*args, **kwargs):
    """
    C++ interface function for renorm_ in dygraph.
    """
def repeat_interleave(*args, **kwargs):
    """
    C++ interface function for repeat_interleave in dygraph.
    """
def requantize(*args, **kwargs):
    """
    C++ interface function for requantize in dygraph.
    """
def reshape(*args, **kwargs):
    """
    C++ interface function for reshape in dygraph.
    """
def reshape2(*args, **kwargs):
    """
    C++ interface function for reshape2 in dygraph.
    """
def reshape2_(*args, **kwargs):
    """
    C++ interface function for reshape2_ in dygraph.
    """
def reshape_(*args, **kwargs):
    """
    C++ interface function for reshape_ in dygraph.
    """
def resnet_basic_block(*args, **kwargs):
    """
    C++ interface function for resnet_basic_block in dygraph.
    """
def resnet_unit(*args, **kwargs):
    """
    C++ interface function for resnet_unit in dygraph.
    """
def reverse(*args, **kwargs):
    """
    C++ interface function for reverse in dygraph.
    """
def rms_norm(*args, **kwargs):
    """
    C++ interface function for rms_norm in dygraph.
    """
def rmsprop(*args, **kwargs):
    """
    C++ interface function for rmsprop in dygraph.
    """
def rnn(*args, **kwargs):
    """
    C++ interface function for rnn in dygraph.
    """
def roformer_relative_embedding_xpu(*args, **kwargs):
    """
    C++ interface function for roformer_relative_embedding_xpu in dygraph.
    """
def roi_align(*args, **kwargs):
    """
    C++ interface function for roi_align in dygraph.
    """
def roi_pool(*args, **kwargs):
    """
    C++ interface function for roi_pool in dygraph.
    """
def roll(*args, **kwargs):
    """
    C++ interface function for roll in dygraph.
    """
def round(*args, **kwargs):
    """
    C++ interface function for round in dygraph.
    """
def round_(*args, **kwargs):
    """
    C++ interface function for round_ in dygraph.
    """
def row_conv(*args, **kwargs):
    """
    C++ interface function for row_conv in dygraph.
    """
def rprop(*args, **kwargs):
    """
    C++ interface function for rprop in dygraph.
    """
def rrelu(*args, **kwargs):
    """
    C++ interface function for rrelu in dygraph.
    """
def rsqrt(*args, **kwargs):
    """
    C++ interface function for rsqrt in dygraph.
    """
def rsqrt_(*args, **kwargs):
    """
    C++ interface function for rsqrt_ in dygraph.
    """
def run_program(*args, **kwargs):
    """
    C++ interface function for run_program in dygraph.
    """
def save(*args, **kwargs):
    """
    C++ interface function for save in dygraph.
    """
def save_combine(*args, **kwargs):
    """
    C++ interface function for save_combine in dygraph.
    """
def scale(*args, **kwargs):
    """
    C++ interface function for scale in dygraph.
    """
def scale_(*args, **kwargs):
    """
    C++ interface function for scale_ in dygraph.
    """
def scatter(*args, **kwargs):
    """
    C++ interface function for scatter in dygraph.
    """
def scatter_(*args, **kwargs):
    """
    C++ interface function for scatter_ in dygraph.
    """
def scatter_nd_add(*args, **kwargs):
    """
    C++ interface function for scatter_nd_add in dygraph.
    """
def searchsorted(*args, **kwargs):
    """
    C++ interface function for searchsorted in dygraph.
    """
def seed(*args, **kwargs):
    """
    C++ interface function for seed in dygraph.
    """
def segment_pool(*args, **kwargs):
    """
    C++ interface function for segment_pool in dygraph.
    """
def self_dp_attention(*args, **kwargs):
    """
    C++ interface function for self_dp_attention in dygraph.
    """
def selu(*args, **kwargs):
    """
    C++ interface function for selu in dygraph.
    """
def sequence_conv(*args, **kwargs):
    """
    C++ interface function for sequence_conv in dygraph.
    """
def sequence_expand(*args, **kwargs):
    """
    C++ interface function for sequence_expand in dygraph.
    """
def sequence_mask(*args, **kwargs):
    """
    C++ interface function for sequence_mask in dygraph.
    """
def sequence_pool(*args, **kwargs):
    """
    C++ interface function for sequence_pool in dygraph.
    """
def sequence_softmax(*args, **kwargs):
    """
    C++ interface function for sequence_softmax in dygraph.
    """
def sequence_unpad_xpu(*args, **kwargs):
    """
    C++ interface function for sequence_unpad_xpu in dygraph.
    """
def set_value(*args, **kwargs):
    """
    C++ interface function for set_value in dygraph.
    """
def set_value_(*args, **kwargs):
    """
    C++ interface function for set_value_ in dygraph.
    """
def sgd(*args, **kwargs):
    """
    C++ interface function for sgd in dygraph.
    """
def shadow_output(*args, **kwargs):
    """
    C++ interface function for shadow_output in dygraph.
    """
def shape(*args, **kwargs):
    """
    C++ interface function for shape in dygraph.
    """
def shard_index(*args, **kwargs):
    """
    C++ interface function for shard_index in dygraph.
    """
def share_buffer(*args, **kwargs):
    """
    C++ interface function for share_buffer in dygraph.
    """
def share_data(*args, **kwargs):
    """
    C++ interface function for share_data in dygraph.
    """
def shuffle_batch(*args, **kwargs):
    """
    C++ interface function for shuffle_batch in dygraph.
    """
def shuffle_channel(*args, **kwargs):
    """
    C++ interface function for shuffle_channel in dygraph.
    """
def sigmoid(*args, **kwargs):
    """
    C++ interface function for sigmoid in dygraph.
    """
def sigmoid_(*args, **kwargs):
    """
    C++ interface function for sigmoid_ in dygraph.
    """
def sigmoid_cross_entropy_with_logits(*args, **kwargs):
    """
    C++ interface function for sigmoid_cross_entropy_with_logits in dygraph.
    """
def sigmoid_cross_entropy_with_logits_(*args, **kwargs):
    """
    C++ interface function for sigmoid_cross_entropy_with_logits_ in dygraph.
    """
def sign(*args, **kwargs):
    """
    C++ interface function for sign in dygraph.
    """
def silu(*args, **kwargs):
    """
    C++ interface function for silu in dygraph.
    """
def sin(*args, **kwargs):
    """
    C++ interface function for sin in dygraph.
    """
def sin_(*args, **kwargs):
    """
    C++ interface function for sin_ in dygraph.
    """
def sine_pos_xpu(*args, **kwargs):
    """
    C++ interface function for sine_pos_xpu in dygraph.
    """
def sinh(*args, **kwargs):
    """
    C++ interface function for sinh in dygraph.
    """
def sinh_(*args, **kwargs):
    """
    C++ interface function for sinh_ in dygraph.
    """
def size(*args, **kwargs):
    """
    C++ interface function for size in dygraph.
    """
def skip_layernorm(*args, **kwargs):
    """
    C++ interface function for skip_layernorm in dygraph.
    """
def slice(*args, **kwargs):
    """
    C++ interface function for slice in dygraph.
    """
def slogdeterminant(*args, **kwargs):
    """
    C++ interface function for slogdeterminant in dygraph.
    """
def soft_relu(*args, **kwargs):
    """
    C++ interface function for soft_relu in dygraph.
    """
def soft_relu_(*args, **kwargs):
    """
    C++ interface function for soft_relu_ in dygraph.
    """
def softmax(*args, **kwargs):
    """
    C++ interface function for softmax in dygraph.
    """
def softmax_(*args, **kwargs):
    """
    C++ interface function for softmax_ in dygraph.
    """
def softmax_with_cross_entropy(*args, **kwargs):
    """
    C++ interface function for softmax_with_cross_entropy in dygraph.
    """
def softmax_with_cross_entropy_(*args, **kwargs):
    """
    C++ interface function for softmax_with_cross_entropy_ in dygraph.
    """
def softplus(*args, **kwargs):
    """
    C++ interface function for softplus in dygraph.
    """
def softshrink(*args, **kwargs):
    """
    C++ interface function for softshrink in dygraph.
    """
def softsign(*args, **kwargs):
    """
    C++ interface function for softsign in dygraph.
    """
def solve(*args, **kwargs):
    """
    C++ interface function for solve in dygraph.
    """
def sparse_attention(*args, **kwargs):
    """
    C++ interface function for sparse_attention in dygraph.
    """
def sparse_momentum(*args, **kwargs):
    """
    C++ interface function for sparse_momentum in dygraph.
    """
def spatial_transformer_resblock_xpu(*args, **kwargs):
    """
    C++ interface function for spatial_transformer_resblock_xpu in dygraph.
    """
def spectral_norm(*args, **kwargs):
    """
    C++ interface function for spectral_norm in dygraph.
    """
def split(*args, **kwargs):
    """
    C++ interface function for split in dygraph.
    """
def sqrt(*args, **kwargs):
    """
    C++ interface function for sqrt in dygraph.
    """
def sqrt_(*args, **kwargs):
    """
    C++ interface function for sqrt_ in dygraph.
    """
def square(*args, **kwargs):
    """
    C++ interface function for square in dygraph.
    """
def squared_l2_norm(*args, **kwargs):
    """
    C++ interface function for squared_l2_norm in dygraph.
    """
def squeeze2(*args, **kwargs):
    """
    C++ interface function for squeeze2 in dygraph.
    """
def squeeze2_(*args, **kwargs):
    """
    C++ interface function for squeeze2_ in dygraph.
    """
def squeeze_excitation_block(*args, **kwargs):
    """
    C++ interface function for squeeze_excitation_block in dygraph.
    """
def stack(*args, **kwargs):
    """
    C++ interface function for stack in dygraph.
    """
def standard_gamma(*args, **kwargs):
    """
    C++ interface function for standard_gamma in dygraph.
    """
def stanh(*args, **kwargs):
    """
    C++ interface function for stanh in dygraph.
    """
def stft(*args, **kwargs):
    """
    C++ interface function for stft in dygraph.
    """
def strided_slice(*args, **kwargs):
    """
    C++ interface function for strided_slice in dygraph.
    """
def sum(*args, **kwargs):
    """
    C++ interface function for sum in dygraph.
    """
def svd(*args, **kwargs):
    """
    C++ interface function for svd in dygraph.
    """
def swiglu(*args, **kwargs):
    """
    C++ interface function for swiglu in dygraph.
    """
def swish(*args, **kwargs):
    """
    C++ interface function for swish in dygraph.
    """
def sync_batch_norm(*args, **kwargs):
    """
    C++ interface function for sync_batch_norm in dygraph.
    """
def take_along_axis(*args, **kwargs):
    """
    C++ interface function for take_along_axis in dygraph.
    """
def tan(*args, **kwargs):
    """
    C++ interface function for tan in dygraph.
    """
def tan_(*args, **kwargs):
    """
    C++ interface function for tan_ in dygraph.
    """
def tanh(*args, **kwargs):
    """
    C++ interface function for tanh in dygraph.
    """
def tanh_(*args, **kwargs):
    """
    C++ interface function for tanh_ in dygraph.
    """
def tanh_shrink(*args, **kwargs):
    """
    C++ interface function for tanh_shrink in dygraph.
    """
def tdm_child(*args, **kwargs):
    """
    C++ interface function for tdm_child in dygraph.
    """
def tdm_sampler(*args, **kwargs):
    """
    C++ interface function for tdm_sampler in dygraph.
    """
def temporal_shift(*args, **kwargs):
    """
    C++ interface function for temporal_shift in dygraph.
    """
def tensor_unfold(*args, **kwargs):
    """
    C++ interface function for tensor_unfold in dygraph.
    """
def thresholded_relu(*args, **kwargs):
    """
    C++ interface function for thresholded_relu in dygraph.
    """
def thresholded_relu_(*args, **kwargs):
    """
    C++ interface function for thresholded_relu_ in dygraph.
    """
def tile(*args, **kwargs):
    """
    C++ interface function for tile in dygraph.
    """
def top_k(*args, **kwargs):
    """
    C++ interface function for top_k in dygraph.
    """
def top_k_v2(*args, **kwargs):
    """
    C++ interface function for top_k_v2 in dygraph.
    """
def top_p_sampling(*args, **kwargs):
    """
    C++ interface function for top_p_sampling in dygraph.
    """
def trace(*args, **kwargs):
    """
    C++ interface function for trace in dygraph.
    """
def transfer_layout(*args, **kwargs):
    """
    C++ interface function for transfer_layout in dygraph.
    """
def transpose(*args, **kwargs):
    """
    C++ interface function for transpose in dygraph.
    """
def transpose2(*args, **kwargs):
    """
    C++ interface function for transpose2 in dygraph.
    """
def triangular_solve(*args, **kwargs):
    """
    C++ interface function for triangular_solve in dygraph.
    """
def tril_indices(*args, **kwargs):
    """
    C++ interface function for tril_indices in dygraph.
    """
def tril_triu(*args, **kwargs):
    """
    C++ interface function for tril_triu in dygraph.
    """
def trilinear_interp_v2(*args, **kwargs):
    """
    C++ interface function for trilinear_interp_v2 in dygraph.
    """
def triu_indices(*args, **kwargs):
    """
    C++ interface function for triu_indices in dygraph.
    """
def trunc(*args, **kwargs):
    """
    C++ interface function for trunc in dygraph.
    """
def trunc_(*args, **kwargs):
    """
    C++ interface function for trunc_ in dygraph.
    """
def truncated_gaussian_random(*args, **kwargs):
    """
    C++ interface function for truncated_gaussian_random in dygraph.
    """
def unbind(*args, **kwargs):
    """
    C++ interface function for unbind in dygraph.
    """
def unfold(*args, **kwargs):
    """
    C++ interface function for unfold in dygraph.
    """
def uniform_random(*args, **kwargs):
    """
    C++ interface function for uniform_random in dygraph.
    """
def uniform_random_batch_size_like(*args, **kwargs):
    """
    C++ interface function for uniform_random_batch_size_like in dygraph.
    """
def uniform_random_inplace(*args, **kwargs):
    """
    C++ interface function for uniform_random_inplace in dygraph.
    """
def uniform_random_inplace_(*args, **kwargs):
    """
    C++ interface function for uniform_random_inplace_ in dygraph.
    """
def unique(*args, **kwargs):
    """
    C++ interface function for unique in dygraph.
    """
def unique_consecutive(*args, **kwargs):
    """
    C++ interface function for unique_consecutive in dygraph.
    """
def unpool(*args, **kwargs):
    """
    C++ interface function for unpool in dygraph.
    """
def unpool3d(*args, **kwargs):
    """
    C++ interface function for unpool3d in dygraph.
    """
def unsqueeze2(*args, **kwargs):
    """
    C++ interface function for unsqueeze2 in dygraph.
    """
def unsqueeze2_(*args, **kwargs):
    """
    C++ interface function for unsqueeze2_ in dygraph.
    """
def unstack(*args, **kwargs):
    """
    C++ interface function for unstack in dygraph.
    """
def update_loss_scaling(*args, **kwargs):
    """
    C++ interface function for update_loss_scaling in dygraph.
    """
def variable_length_memory_efficient_attention(*args, **kwargs):
    """
    C++ interface function for variable_length_memory_efficient_attention in dygraph.
    """
def view_dtype(*args, **kwargs):
    """
    C++ interface function for view_dtype in dygraph.
    """
def view_shape(*args, **kwargs):
    """
    C++ interface function for view_shape in dygraph.
    """
def viterbi_decode(*args, **kwargs):
    """
    C++ interface function for viterbi_decode in dygraph.
    """
def warpctc(*args, **kwargs):
    """
    C++ interface function for warpctc in dygraph.
    """
def warprnnt(*args, **kwargs):
    """
    C++ interface function for warprnnt in dygraph.
    """
def weight_dequantize(*args, **kwargs):
    """
    C++ interface function for weight_dequantize in dygraph.
    """
def weight_only_linear(*args, **kwargs):
    """
    C++ interface function for weight_only_linear in dygraph.
    """
def weight_only_linear_xpu(*args, **kwargs):
    """
    C++ interface function for weight_only_linear_xpu in dygraph.
    """
def weight_quantize(*args, **kwargs):
    """
    C++ interface function for weight_quantize in dygraph.
    """
def weighted_sample_neighbors(*args, **kwargs):
    """
    C++ interface function for weighted_sample_neighbors in dygraph.
    """
def where(*args, **kwargs):
    """
    C++ interface function for where in dygraph.
    """
def where_(*args, **kwargs):
    """
    C++ interface function for where_ in dygraph.
    """
def where_index(*args, **kwargs):
    """
    C++ interface function for where_index in dygraph.
    """
def yolo_box(*args, **kwargs):
    """
    C++ interface function for yolo_box in dygraph.
    """
def yolo_box_head(*args, **kwargs):
    """
    C++ interface function for yolo_box_head in dygraph.
    """
def yolo_box_post(*args, **kwargs):
    """
    C++ interface function for yolo_box_post in dygraph.
    """
def yolo_box_xpu(*args, **kwargs):
    """
    C++ interface function for yolo_box_xpu in dygraph.
    """
def yolov3_loss(*args, **kwargs):
    """
    C++ interface function for yolov3_loss in dygraph.
    """

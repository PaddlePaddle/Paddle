# This file records the Unity Build compilation rules.
# The source files in a `register_unity_group` called are compiled in a unity
# file.
# Generally, the combination rules in this file do not need to be modified.
# If there are some redefined error in compiling with the source file which
# in combination rule, you can remove the source file from the following rules.
register_unity_group(cc
    add_position_encoding_op.cc
    addmm_op.cc
    affine_channel_op.cc
    affine_grid_op.cc
    allclose_op.cc
    argsort_op.cc
    array_to_lod_tensor_op.cc
    assert_op.cc
    assign_op.cc
    assign_value_op.cc
    attention_lstm_op.cc
    average_accumulates_op.cc
    batch_fc_op.cc
    bce_loss_op.cc
    beam_search_op.cc
    beam_search_decode_op.cc
    bernoulli_op.cc
    bilateral_slice_op.cc
    activation_cudnn_op.cu.cc
    mkldnn/activation_mkldnn_op.cc
    assign_value_op.cu.cc
    conj_op.cc
    conv_op.cc
    conv_op.cu.cc)
register_unity_group(cc
    mkldnn/batch_norm_mkldnn_op.cc
    bilinear_tensor_product_op.cc
    bmm_op.cc
    bpr_loss_op.cc
    cast_op.cc
    cholesky_op.cc
    chunk_eval_op.cc
    clip_by_norm_op.cc
    clip_op.cc
    coalesce_tensor_op.cc
    arg_min_op.cc
    arg_max_op.cc)
register_unity_group(cc
    center_loss_op.cc
    mkldnn/concat_mkldnn_op.cc
    mkldnn/conv_mkldnn_op.cc
    mkldnn/conv_transpose_mkldnn_op.cc
    correlation_op.cc
    cos_sim_op.cc
    crf_decoding_op.cc
    crop_op.cc
    conv_transpose_op.cc
    expand_as_op.cc
    gru_unit_op.cc)
register_unity_group(cc
    cross_entropy_op.cc
    cross_op.cc
    ctc_align_op.cc
    cudnn_lstm_op.cc
    cumsum_op.cc
    cvm_op.cc
    data_norm_op.cc
    deformable_conv_op.cc
    deformable_conv_v1_op.cc
    deformable_psroi_pooling_op.cc
    delete_var_op.cc
    dequantize_abs_max_op.cc
    dequantize_op.cc
    mkldnn/dequantize_mkldnn_op.cc
    crop_tensor_op.cc
    expand_as_v2_op.cc
    imag_op.cc
    kldiv_loss_op.cc)
register_unity_group(cc
    dequeue_op.cc
    detection_map_op.cc
    dgc_clip_by_norm_op.cc
    diag_embed_op.cc
    diag_op.cc
    diag_v2_op.cc
    dot_op.cc
    edit_distance_op.cc
    empty_op.cc
    enqueue_op.cc
    erf_op.cc
    linear_chain_crf_op.cc
    merge_selected_rows_op.cu.cc
    partial_concat_op.cc
    mkldnn/pool_mkldnn_op.cc
    real_op.cc
    mkldnn/softmax_mkldnn_op.cc
    softmax_with_cross_entropy_op.cc
    squared_l2_distance_op.cc
    top_k_op.cc
    dist_op.cc)
register_unity_group(cc
    expand_v2_op.cc
    fake_dequantize_op.cc
    fc_op.cc
    mkldnn/fc_mkldnn_op.cc
    fill_any_like_op.cc
    fill_constant_batch_size_like_op.cc
    fill_constant_op.cc
    fill_op.cc
    fill_zeros_like_op.cc
    filter_by_instag_op.cc
    rnn_op.cu.cc
    batch_norm_op.cc)
register_unity_group(cc
    flatten_op.cc
    flip_op.cc
    fsp_op.cc
    gather_nd_op.cc
    gather_op.cc
    gather_tree_op.cc
    gaussian_random_batch_size_like_op.cc
    gaussian_random_op.cc
    mkldnn/gaussian_random_mkldnn_op.cc
    grid_sampler_op.cc
    group_norm_op.cc
    gru_op.cc
    split_op.cu.cc
    affine_grid_cudnn_op.cu.cc
    beam_search_op.cu.cc
    cudnn_lstm_op.cu.cc
    empty_op.cu.cc
    fc_op.cu.cc
    fill_constant_batch_size_like_op.cu.cc
    fill_constant_op.cu.cc
    fill_op.cu.cc
    fill_zeros_like_op.cu.cc
    flatten_op.cu.cc
    grid_sampler_cudnn_op.cu.cc
    gru_op.cu.cc
    inverse_op.cu.cc
    is_empty_op.cu.cc
    maxout_op.cu.cc
    mul_op.cu.cc
    concat_op.cu.cc
    mul_op.cu.cc
    pool_op.cu.cc
    pool_cudnn_op.cu.cc
    pool_with_index_op.cu.cc
    run_program_op.cu.cc
    softmax_op.cu.cc
    softmax_cudnn_op.cu.cc
    spp_op.cu.cc
    squeeze_op.cu.cc
    unbind_op.cu.cc
    unique_op.cu
    unpool_op.cu.cc
    unsqueeze_op.cu.cc)
register_unity_group(cc
    hash_op.cc
    hierarchical_sigmoid_op.cc
    hinge_loss_op.cc
    histogram_op.cc
    huber_loss_op.cc
    im2sequence_op.cc
    increment_op.cc
    index_sample_op.cc
    index_select_op.cc
    interpolate_op.cc
    isfinite_v2_op.cc
    smooth_l1_loss_op.cc
    uniform_random_batch_size_like_op.cc
    uniform_random_op.cc
    unique_op.cc
    unique_with_counts_op.cc
    unpool_op.cc
    unsqueeze_op.cc
    unstack_op.cc
    var_conv_2d_op.cc
    where_index_op.cc
    where_op.cc)
register_unity_group(cc
    inplace_abn_op.cc
    interpolate_v2_op.cc
    inverse_op.cc
    is_empty_op.cc
    isfinite_op.cc
    kron_op.cc
    l1_norm_op.cc
    label_smooth_op.cc
    layer_norm_op.cc
    mkldnn/layer_norm_mkldnn_op.cc
    mkldnn/layer_norm_mkldnn_op.cc
    linspace_op.cc
    load_combine_op.cc
    load_op.cc
    row_conv_op.cc
    tensor_array_to_tensor_op.cc
    tile_op.cc
    top_k_v2_op.cc
    trace_op.cc)
register_unity_group(cc
    lod_array_length_op.cc
    lod_rank_table_op.cc
    lod_reset_op.cc
    lod_tensor_to_array_op.cc
    log_softmax_op.cc
    lookup_table_dequant_op.cc
    lrn_op.cc
    mkldnn/lrn_mkldnn_op.cc
    lstm_unit_op.cc
    lstmp_op.cc
    transpose_op.cc
    mkldnn/transpose_mkldnn_op.cc
    tree_conv_op.cc
    tril_triu_op.cc
    truncated_gaussian_random_op.cc
    unbind_op.cc
    unfold_op.cc
    space_to_depth_op.cc
    spectral_norm_op.cc
    split_op.cc
    split_selected_rows_op.cc
    spp_op.cc
    squared_l2_norm_op.cc
    squeeze_op.cc
    stack_op.cc)
register_unity_group(cc
    log_loss_op.cc
    lookup_table_v2_op.cc
    margin_rank_loss_op.cc
    masked_select_op.cc
    match_matrix_tensor_op.cc
    matmul_op.cc
    mkldnn/matmul_mkldnn_op.cc
    max_sequence_len_op.cc
    maxout_op.cc
    merge_lod_tensor_op.cc
    merge_selected_rows_op.cc
    meshgrid_op.cc
    sum_op.cc
    mkldnn/sum_mkldnn_op.cc
    tdm_child_op.cc
    tdm_sampler_op.cc
    teacher_student_sigmoid_loss_op.cc
    temporal_shift_op.cc)
register_unity_group(cc
    concat_op.cc
    conv_shift_op.cc
    dequantize_log_op.cc
    dropout_op.cc
    expand_op.cc
    fake_quantize_op.cc
    gelu_op.cc
    get_tensor_from_selected_rows_op.cc
    lookup_table_op.cc
    matmul_v2_op.cc)
register_unity_group(cc
    mean_iou_op.cc
    mean_op.cc
    minus_op.cc
    mish_op.cc
    mul_op.cc
    multinomial_op.cc
    multiplex_op.cc
    mv_op.cc
    nce_op.cc
    nll_loss_op.cc
    norm_op.cc
    one_hot_op.cc
    one_hot_v2_op.cc
    p_norm_op.cc
    pad2d_op.cc
    pad3d_op.cc
    pad_constant_like_op.cc
    pad_op.cc
    split_lod_tensor_op.cc
    roi_pool_op.cc
    selu_op.cc
    shape_op.cc
    shard_index_op.cc)
register_unity_group(cc
    modified_huber_loss_op.cc
    mkldnn/mul_mkldnn_op.cc
    partial_sum_op.cc
    pixel_shuffle_op.cc
    pool_op.cc
    pool_with_index_op.cc
    positive_negative_pair_op.cc
    prelu_op.cc
    print_op.cc
    prroi_pool_op.cc
    psroi_pool_op.cc
    pull_box_extended_sparse_op.cc
    pull_box_sparse_op.cc
    pull_sparse_op.cc
    pull_sparse_v2_op.cc
    strided_slice_op.cc)
register_unity_group(cc
    push_dense_op.cc
    quantize_op.cc
    mkldnn/quantize_mkldnn_op.cc
    queue_generator_op.cc
    randint_op.cc
    random_crop_op.cc
    randperm_op.cc
    range_op.cc
    rank_attention_op.cc
    rank_loss_op.cc
    recurrent_op.cc
    reorder_lod_tensor_by_rank_op.cc
    requantize_op.cc
    mkldnn/requantize_mkldnn_op.cc
    reshape_op.cc
    reverse_op.cc
    shrink_rnn_memory_op.cc
    shuffle_batch_op.cc
    shuffle_channel_op.cc
    sigmoid_cross_entropy_with_logits_op.cc
    sign_op.cc
    similarity_focus_op.cc
    size_op.cc
    softmax_op.cc)
register_unity_group(cc
    rnn_memory_helper_op.cc
    roi_align_op.cc
    roll_op.cc
    run_program_op.cc
    sample_logits_op.cc
    sampling_id_op.cc
    save_combine_op.cc
    save_op.cc
    scale_op.cc
    scatter_nd_add_op.cc
    scatter_op.cc
    seed_op.cc
    segment_pool_op.cc
    select_input_op.cc
    select_output_op.cc
    slice_op.cc)
register_unity_group(cu
    addmm_op.cu
    affine_channel_op.cu
    allclose_op.cu
    argsort_op.cu
    assign_value_op.cu
    bce_loss_op.cu
    bernoulli_op.cu
    bilateral_slice_op.cu)
register_unity_group(cu
    bilinear_tensor_product_op.cu
    bmm_op.cu
    cast_op.cu
    cholesky_op.cu
    clip_by_norm_op.cu
    clip_op.cu
    affine_grid_op.cu
    average_accumulates_op.cu
    conj_op.cu
    conv_cudnn_op.cu
    correlation_op.cu
    deformable_psroi_pooling_op.cu
    dot_op.cu
    imag_op.cu
    lstmp_op.cu
    real_op.cu
    softmax_cudnn_op.cu
    sigmoid_cross_entropy_with_logits_op.cu
    softmax_with_cross_entropy_op.cu)
register_unity_group(cu
    center_loss_op.cu
    conv_op.cu
    conv_transpose_cudnn_op.cu
    conv_transpose_op.cu
    cos_sim_op.cu
    crop_op.cu
    arg_min_op.cu
    arg_max_op.cu
    squared_l2_distance_op.cu
    one_hot_v2_op.cu
    shuffle_channel_op.cu
    gru_unit_op.cu)
register_unity_group(cu
    cross_entropy_op.cu
    cross_op.cu
    ctc_align_op.cu
    cumsum_op.cu
    cvm_op.cu
    data_norm_op.cu
    deformable_conv_op.cu
    deformable_conv_v1_op.cu
    dequantize_abs_max_op.cu
    top_k_op.cu)
register_unity_group(cu
    dgc_clip_by_norm_op.cu
    diag_embed_op.cu
    diag_op.cu
    diag_v2_op.cu
    edit_distance_op.cu
    erf_op.cu
    top_k_v2_op.cu)
register_unity_group(cu
    expand_v2_op.cu
    fake_dequantize_op.cu
    fill_any_like_op.cu
    psroi_pool_op.cu
    rank_loss_op.cu
    instance_norm_op.cu)
register_unity_group(cu
    flip_op.cu
    fsp_op.cu
    gather_nd_op.cu
    gather_op.cu
    gather_tree_op.cu
    gaussian_random_op.cu
    grid_sampler_op.cu
    group_norm_op.cu
    expand_op.cu)
register_unity_group(cu
    hinge_loss_op.cu
    histogram_op.cu
    huber_loss_op.cu
    im2sequence_op.cu
    increment_op.cu
    index_sample_op.cu
    index_select_op.cu
    interpolate_op.cu
    isfinite_v2_op.cu
    expand_as_op.cu
    partial_concat_op.cu
    kldiv_loss_op.cu
    meshgrid_op.cu)
register_unity_group(cu
    inplace_abn_op.cu
    interpolate_v2_op.cu
    isfinite_op.cu
    kron_op.cu
    l1_norm_op.cu
    label_smooth_op.cu
    layer_norm_op.cu
    linspace_op.cu
    load_combine_op.cu
    load_op.cu)
register_unity_group(cu
    lod_reset_op.cu
    log_softmax_op.cu
    lrn_op.cu
    lstm_unit_op.cu
    expand_as_v2_op.cu
    batch_norm_op.cu)
register_unity_group(cu
    log_loss_op.cu
    lookup_table_v2_op.cu
    margin_rank_loss_op.cu
    masked_select_op.cu
    merge_selected_rows_op.cu
    crop_tensor_op.cu
    unstack_op.cu
    where_index_op.cu
    where_op.cu
    uniform_random_op.cu
    tree_conv_op.cu
    tril_triu_op.cu
    truncated_gaussian_random_op.cu
    unfold_op.cu
    row_conv_op.cu
    transpose_op.cu)
register_unity_group(cu
    conv_shift_op.cu
    dequantize_log_op.cu
    dropout_op.cu
    fake_quantize_op.cu
    gelu_op.cu
    lookup_table_op.cu
    smooth_l1_loss_op.cu
    space_to_depth_op.cu
    spectral_norm_op.cu
    split_selected_rows_op.cu
    squared_l2_norm_op.cu
    stack_op.cu
    roi_pool_op.cu
    selu_op.cu
    shape_op.cu
    shard_index_op.cu
    sign_op.cu)
register_unity_group(cu
    mean_iou_op.cu
    mean_op.cu
    minus_op.cu
    mish_op.cu
    multinomial_op.cu
    multiplex_op.cu
    mv_op.cu
    nll_loss_op.cu
    norm_op.cu
    one_hot_op.cu
    p_norm_op.cu
    pad2d_op.cu
    pad3d_op.cu
    pad_constant_like_op.cu
    pad_op.cu
    sum_op.cu
    temporal_shift_op.cu
    size_op.cu)
register_unity_group(cu
    partial_sum_op.cu
    pixel_shuffle_op.cu
    prelu_op.cu
    prroi_pool_op.cu
    pull_box_extended_sparse_op.cu
    pull_box_sparse_op.cu)
register_unity_group(cu
    randint_op.cu
    random_crop_op.cu
    randperm_op.cu
    range_op.cu
    reverse_op.cu
    strided_slice_op.cu)
register_unity_group(cu
    roi_align_op.cu
    roll_op.cu
    sample_logits_op.cu
    sampling_id_op.cu
    save_combine_op.cu
    save_op.cu
    scale_op.cu
    scatter_nd_add_op.cu
    scatter_op.cu
    seed_op.cu
    segment_pool_op.cu
    slice_op.cu)
register_unity_group(cu
    batch_fc_op.cu
    matmul_v2_op.cu)
register_unity_group(cu
    trace_op.cu
    rank_attention_op.cu)
# The following groups are to make better use of `/MP` which MSVC's parallel
# compilation instruction when compiling in Unity Build.
register_unity_group(cu tile_op.cu)
register_unity_group(cu unique_op.cu)
register_unity_group(cu activation_op.cu)
register_unity_group(cu dist_op.cu)

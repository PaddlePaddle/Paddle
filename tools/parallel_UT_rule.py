# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os

# *=======These unittest doesn't occupy GPU memory, just run as CPU unittest=======* #
# It run 8 job each time, If it failed due to Insufficient GPU memory or CUBLAS_STATUS_ALLOC_FAILED, 
# just remove it from this list.
CPU_PARALLEL_JOB = [
    'test_row_conv',
    'test_nce',
    'test_conv3d_mkldnn_op',
    'dim_test',
    'test_limit_gpu_memory',
    'profiler_test',
    'test_dequantize_mkldnn_op',
    'test_elementwise_add_bf16_mkldnn_op',
    'test_rpn_target_assign_op',
    'test_hash_op',
    'reader_blocking_queue_test',
    'jit_kernel_test',
    'test_tdm_child_op',
    'test_simplify_with_basic_ops_pass',
    'test_sequence_last_step',
    'test_sequence_first_step',
    'test_seq_concat_fc_fuse_pass',
    'test_fc_gru_fuse_pass',
    'test_dataset_imdb',
    'dlpack_tensor_test',
    'check_reduce_rank_test',
    'var_type_traits_test',
    'var_type_inference_test',
    'to_string_test',
    'threadpool_test',
    'test_version',
    'test_var_info',
    'test_var_conv_2d',
    'test_unique_name',
    'test_transpose_int8_mkldnn_op',
    'test_transpose_bf16_mkldnn_op',
    'test_trainable',
    'test_teacher_student_sigmoid_loss_op',
    'test_tdm_sampler_op',
    'test_switch',
    'test_static_shape_inferrence_for_shape_tensor',
    'test_squared_mat_sub_fuse_pass',
    'test_sequence_scatter_op',
    'test_sequence_scatter_op',
    'test_scaled_dot_product_attention',
    'test_rnn_memory_helper_op',
    'test_requantize_mkldnn_op',
    'test_quantize_transpiler',
    'test_quantize_mkldnn_op',
    'test_py_reader_sample_generator',
    'test_parallel_executor_seresnext_with_reduce_cpu',
    'test_parallel_executor_seresnext_with_fuse_all_reduce_cpu',
    'test_parallel_executor_seresnext_base_cpu',
    'test_parallel_dygraph_sync_batch_norm',
    'test_origin_info',
    'test_multiclass_nms_op',
    'test_mkldnn_conv_bias_fuse_pass',
    'test_mkldnn_conv_activation_fuse_pass',
    'test_matrix_nms_op',
    'test_ir_graph',
    'test_inference_api',
    'test_infer_shape',
    'test_infer_no_need_buffer_slots',
    'test_imperative_numpy_bridge',
    'test_imperative_decorator',
    'test_hooks',
    'test_gpu_package_without_gpu_device',
    'test_global_var_getter_setter',
    'test_get_set_flags',
    'test_fusion_repeated_fc_relu_op',
    'test_fused_emb_seq_pool_op',
    'test_fleet_base_4',
    'test_fc_lstm_fuse_pass',
    'test_executor_feed_non_tensor',
    'test_executor_check_feed',
    'test_executor_and_use_program_cache',
    'test_exception',
    'test_error_clip',
    'test_embedding_eltwise_layernorm_fuse_pass',
    'test_dyn_rnn',
    'test_dpsgd_op',
    'test_distributed_reader',
    'test_directory_migration',
    'test_dataset_wmt',
    'test_dataset_uci_housing',
    'test_dataset_cifar',
    'test_data_feeder',
    'test_cudnn_placement_pass',
    'test_conv3d_layer',
    'test_concat_bf16_mkldnn_op',
    'test_common_infer_shape_functions',
    'test_check_import_scipy',
    'test_calc_gradient',
    'test_bipartite_match_op',
    'test_attention_lstm_op',
    'test_array_read_write_op',
    'stringprintf_test',
    'stringpiece_test',
    'selected_rows_test',
    'scope_test',
    'reader_test',
    'prune_test',
    'op_tester',
    'eigen_test',
    'device_worker_test',
    'cudnn_helper_test',
    'cudnn_desc_test',
    'tuple_test',
    'timer_test',
    'test_zeros_op',
    'test_while_op',
    'test_utils',
    'test_static_analysis',
    'test_split_and_merge_lod_tensor_op',
    'test_spawn_and_init_parallel_env',
    'test_slice_var',
    'test_similarity_focus_op',
    'test_shuffle_batch_op',
    'test_shrink_rnn_memory',
    'test_set_bool_attr',
    'test_sequence_topk_avg_pooling',
    'test_selected_rows',
    'test_scope',
    'test_sampling_id_op',
    'test_runtime_and_compiletime_exception',
    'test_run_fluid_by_module_or_command_line',
    'test_retinanet_detection_output',
    'test_require_version',
    'test_repeated_fc_relu_fuse_pass',
    'test_registry',
    'test_recurrent_op',
    'test_recommender_system',
    'test_query_op',
    'test_quantization_mkldnn_pass',
    'test_quant2_int8_mkldnn_pass',
    'test_pybind_interface',
    'test_py_reader_error_msg',
    'test_prune',
    'test_protobuf',
    'test_progressbar',
    'test_program_to_string',
    'test_program_code',
    'test_program',
    'test_precision_recall_op',
    'test_positive_negative_pair_op',
    'test_parallel_executor_run_load_infer_program',
    'test_op_version',
    'test_op_support_gpu',
    'test_ones_op',
    'test_npair_loss_op',
    'test_nn_functional_embedding_static',
    'test_name_scope',
    'test_multiprocess_dataloader_iterable_dataset_split',
    'test_multi_gru_mkldnn_op',
    'test_mul_int8_mkldnn_op',
    'test_mkldnn_scale_matmul_fuse_pass',
    'test_mkldnn_op_inplace',
    'test_mkldnn_matmul_transpose_reshape_fuse_pass',
    'test_mkldnn_inplace_fuse_pass',
    'test_mkldnn_cpu_bfloat16_pass',
    'test_mine_hard_examples_op',
    'test_memory_usage',
    'test_matmul_mkldnn_op',
    'test_matmul_bf16_mkldnn_op',
    'test_math_op_patch',
    'test_match_matrix_tensor_op',
    'test_lookup_table_dequant_op',
    'test_logging_utils',
    'test_logger',
    'test_lod_tensor_array_ops',
    'test_lod_tensor_array',
    'test_lod_rank_table',
    'test_lod_array_length_op',
    'test_locality_aware_nms_op',
    'test_load_vars_shape_check',
    'test_load_op_xpu',
    'test_load_op',
    'test_linear_chain_crf_op',
    'test_layer_norm_mkldnn_op',
    'test_layer_norm_bf16_mkldnn_op',
    'test_lambv2_op',
    'test_ir_skip_layernorm_pass',
    'test_io_save_load',
    'test_input_spec',
    'test_inference_model_io',
    'test_imperative_base',
    'test_image_classification_layer',
    'test_image',
    'test_ifelse_basic',
    'test_hsigmoid_op',
    'test_generator',
    'test_generate_proposal_labels_op',
    'test_generate_mask_labels_op',
    'test_gast_with_compatibility',
    'test_fusion_squared_mat_sub_op',
    'test_fusion_seqconv_eltadd_relu_op',
    'test_fusion_lstm_op',
    'test_fusion_gru_op',
    'test_fusion_gru_int8_mkldnn_op',
    'test_fusion_gru_bf16_mkldnn_op',
    'test_fused_embedding_fc_lstm_op',
    'test_function_spec',
    'test_full_op',
    'test_framework_debug_str',
    'test_fp16_utils',
    'test_fleet_rolemaker_4',
    'test_flags_use_mkldnn',
    'test_filter_by_instag_op',
    'test_fetch_var',
    'test_fetch_handler',
    'test_feed_fetch_method',
    'test_fc_mkldnn_op',
    'test_fc_lstm_fuse_pass',
    'test_fc_gru_fuse_pass',
    'test_fc_bf16_mkldnn_op',
    'test_entry_attr',
    'test_entry_attr2',
    'test_elementwise_mul_bf16_mkldnn_op',
    'test_eager_deletion_recurrent_op',
    'test_eager_deletion_padding_rnn',
    'test_eager_deletion_mnist',
    'test_eager_deletion_dynamic_rnn_base',
    'test_eager_deletion_conditional_block',
    'test_dynrnn_static_input',
    'test_dynrnn_gradient_check',
    'test_dygraph_mode_of_unittest',
    'test_download',
    'test_distributions',
    'test_detection_map_op',
    'test_desc_clone',
    'test_depthwise_conv_mkldnn_pass',
    'test_deprecated_memory_optimize_interfaces',
    'test_default_scope_funcs',
    'test_default_dtype',
    'test_dataset_voc',
    'test_dataset_movielens',
    'test_dataset_imikolov',
    'test_dataset_conll05',
    'test_data_generator',
    'test_data',
    'test_cyclic_cifar_dataset',
    'test_crypto',
    'test_create_op_doc_string',
    'test_create_global_var',
    'test_conv3d_transpose_layer',
    'test_conv2d_transpose_layer',
    'test_conv2d_mkldnn_op',
    'test_conv2d_layer',
    'test_conv2d_int8_mkldnn_op',
    'test_conv2d_bf16_mkldnn_op',
    'test_const_value',
    'test_conditional_block',
    'test_concat_int8_mkldnn_op',
    'test_compat',
    'test_collective_base',
    'test_collective_api_base',
    'test_chunk_eval_op',
    'test_broadcast_to_op',
    'test_broadcast_shape',
    'test_broadcast_error',
    'test_bpr_loss_op',
    'test_beam_search_op',
    'test_batch_sampler',
    'test_basic_rnn_name',
    'test_aligned_allocator',
    'scatter_test',
    'save_load_combine_op_test',
    'program_desc_test',
    'lodtensor_printer_test',
    'lod_tensor_test',
    'gather_test',
    'gather_op_test',
    'fused_broadcast_op_test',
    'exception_holder_test',
    'decorator_test',
    'ddim_test',
    'data_layout_transform_test',
    'cpu_vec_test',
    'cow_ptr_tests',
    'conditional_block_op_test',
    'bfloat16_test',
    'assign_op_test',
    'unroll_array_ops_test',
    'test_seqpool_cvm_concat_fuse_pass',
    'test_seqpool_concat_fuse_pass',
    'test_reshape_bf16_op',
    'test_repeated_fc_relu_fuse_pass',
    'test_py_reader_return_list',
    'test_py_reader_lod_level_share',
    'test_protobuf_descs',
    'test_paddle_inference_api',
    'test_operator_desc',
    'test_operator',
    'test_mkldnn_matmul_op_output_fuse_pass',
    'test_mkldnn_inplace_pass',
    'test_mkldnn_conv_concat_relu_mkldnn_fuse_pass',
    'test_layer',
    'test_is_test_pass',
    'test_graph_pattern_detector',
    'test_fusion_seqpool_cvm_concat_op',
    'test_fusion_seqpool_concat_op',
    'test_fusion_seqexpand_concat_fc_op',
    'test_fusion_gru_mkldnn_op',
    'test_fleet_util',
    'test_fleet_runtime',
    'test_fleet_rolemaker_init',
    'test_flags_mkldnn_ops_on_off',
    'test_dataset_download',
    'test_dataloader_unkeep_order',
    'test_dataloader_keep_order',
    'test_dataloader_dataset',
    'test_crf_decoding_op',
    'test_create_parameter',
    'test_context_manager',
    'test_analyzer',
    'tensor_test',
    'split_test',
    'save_load_op_test',
    'place_test',
    'op_version_registry_test',
    'op_proto_maker_test',
    'op_kernel_type_test',
    'mask_util_test',
    'inlined_vector_test',
    'infer_io_utils_tester',
    'errors_test',
    'enforce_test',
    'dropout_op_test',
    'data_type_test',
    'cpu_info_test',
    'cpu_helper_test',
    'beam_search_decode_op_test',
    'auto_growth_best_fit_allocator_test',
    'test_skip_layernorm_fuse_pass',
    'test_multihead_matmul_fuse_pass',
    'test_fc_elementwise_layernorm_fuse_pass',
    'version_test',
    'variable_test',
    'test_scale_matmul_fuse_pass',
    'test_reshape_transpose_matmul_mkldnn_fuse_pass',
    'test_multi_gru_seq_fuse_pass',
    'test_multi_gru_fuse_pass',
    'test_mkldnn_placement_pass',
    'test_mkldnn_op_nhwc',
    'test_matmul_transpose_reshape_fuse_pass',
    'test_fs',
    'test_fleet',
    'test_cpu_quantize_squash_pass',
    'test_cpu_quantize_placement_pass',
    'test_cpu_quantize_pass',
    'test_cpu_bfloat16_placement_pass',
    'test_cpu_bfloat16_pass',
    'test_conv_elementwise_add_mkldnn_fuse_pass',
    'test_conv_concat_relu_mkldnn_fuse_pass',
    'test_conv_bias_mkldnn_fuse_pass',
    'test_conv_batch_norm_mkldnn_fuse_pass',
    'test_conv_activation_mkldnn_fuse_pass',
    'test_benchmark',
    'test_batch_norm_act_fuse_pass',
    'selected_rows_functor_test',
    'save_load_util_test',
    'pass_test',
    'operator_test',
    'operator_exception_test',
    'op_debug_string_test',
    'op_compatible_info_test',
    'op_call_stack_test',
    'node_test',
    'no_need_buffer_vars_inference_test',
    'nccl_context_test',
    'math_function_test',
    'init_test',
    'graph_to_program_pass_test',
    'graph_test',
    'graph_helper_test',
    'float16_test',
    'dist_multi_trainer_test',
    'cipher_utils_test',
    'broadcast_op_test',
    'aes_cipher_test',
]

# It run 4 job each time, If it failed due to Insufficient GPU memory or CUBLAS_STATUS_ALLOC_FAILED, 
# just remove it from this list.
TETRAD_PARALLEL_JOB = [
    'system_allocator_test',
    'buffered_allocator_test',
    'test_tensor_to_numpy',
    'test_imperative_framework',
    'test_naive_best_fit_gpu_memory_limit',
    'test_auto_growth_gpu_memory_limit',
    'test_imperative_using_non_zero_gpu',
    'cuda_helper_test',
    'retry_allocator_test',
    'allocator_facade_frac_flags_test',
]


def main():
    eight_parallel_job = '^job$'
    tetrad_parallel_job = '^job$'
    non_parallel_job_1 = '^job$'
    non_parallel_job_2 = '^job$'

    test_cases = sys.argv[1]
    test_cases = test_cases.split("\n")
    for unittest in test_cases:
        if unittest in CPU_PARALLEL_JOB:
            eight_parallel_job = eight_parallel_job + '|^' + unittest + '$'
            continue
        if unittest in TETRAD_PARALLEL_JOB:
            tetrad_parallel_job = tetrad_parallel_job + '|^' + unittest + '$'
            continue

        if len(non_parallel_job_1) < 10000:
            non_parallel_job_1 = non_parallel_job_1 + '|^' + unittest + '$'
        else:
            non_parallel_job_2 = non_parallel_job_2 + '|^' + unittest + '$'

    non_parallel_job = ",".join([non_parallel_job_1, non_parallel_job_2])
    print("{};{};{}".format(eight_parallel_job, tetrad_parallel_job,
                            non_parallel_job))


if __name__ == '__main__':
    main()

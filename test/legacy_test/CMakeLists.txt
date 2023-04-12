file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

file(
  GLOB DIST_TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_dist_*.py")
string(REPLACE ".py" "" DIST_TEST_OPS "${DIST_TEST_OPS}")

foreach(TEST_OP ${DIST_TEST_OPS})
  list(REMOVE_ITEM TEST_OPS ${TEST_OP})
endforeach()

if(NOT WITH_COVERAGE)
  list(REMOVE_ITEM TEST_OPS test_hapi_hub)
endif()

foreach(src ${TEST_OPS})
  py_test(${src} SRCS ${src}.py)
endforeach()

function(py_dist_test TARGET_NAME)
  if(WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS ENVS)
    cmake_parse_arguments(py_dist_test "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    if(WITH_COVERAGE
       AND (WITH_GPU OR WITH_ROCM)
       AND (WITH_NCCL OR WITH_RCCL)
       AND NOT WIN32)
      add_test(
        NAME ${TARGET_NAME}
        COMMAND
          ${CMAKE_COMMAND} -E env FLAGS_init_allocated_mem=true
          FLAGS_cudnn_deterministic=true FLAGS_cpu_deterministic=true
          NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
          PYTHONPATH=${PADDLE_BINARY_DIR}/python ${py_dist_test_ENVS}
          COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
          ${PYTHON_EXECUTABLE} -u ${py_dist_test_SRCS} ${py_dist_test_ARGS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      # No unit test should exceed 10 minutes.
      set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 600 LABELS
                                                     "RUN_TYPE=DIST")
    endif()

  endif()
endfunction()

foreach(src ${DIST_TEST_OPS})
  message(STATUS ${src})
  py_dist_test(${src} SRCS ${src}.py)
endforeach()
set_tests_properties(test_dataset_cifar PROPERTIES TIMEOUT 120)
set_tests_properties(test_pretrained_model PROPERTIES TIMEOUT 120)
set_tests_properties(test_model PROPERTIES TIMEOUT 300)
set_tests_properties(test_dataset_movielens PROPERTIES TIMEOUT 120)
set_tests_properties(test_datasets PROPERTIES TIMEOUT 300)
set_tests_properties(test_dataset_wmt PROPERTIES TIMEOUT 120)
set_tests_properties(test_vision_models PROPERTIES TIMEOUT 120)
set_tests_properties(test_dataset_uci_housing PROPERTIES TIMEOUT 120)
set_tests_properties(test_dataset_imdb PROPERTIES TIMEOUT 300)
set_tests_properties(test_pretrained_model PROPERTIES TIMEOUT 600)
set_tests_properties(test_callback_wandb PROPERTIES TIMEOUT 60)
if(WITH_COVERAGE)
  set_tests_properties(test_hapi_hub PROPERTIES TIMEOUT 300)
endif()

if(APPLE)
  set_tests_properties(test_callback_early_stop PROPERTIES TIMEOUT 300)
  set_tests_properties(test_callback_reduce_lr_on_plateau PROPERTIES TIMEOUT
                                                                     300)
  set_tests_properties(test_vision_models PROPERTIES TIMEOUT 300)
endif()
set_tests_properties(test_decorator PROPERTIES TIMEOUT 120)


set_tests_properties(
  test_buffer_shared_memory_reuse_pass_and_fuse_optimization_op_pass
  test_data_norm_op test_dataloader_keep_order test_dataloader_unkeep_order
  test_buffer_shared_memory_reuse_pass PROPERTIES LABELS "RUN_TYPE=DIST")
set_tests_properties(
  test_sync_batch_norm_op
  test_inplace_abn_op
  test_parallel_executor_seresnext_base_gpu
  test_parallel_executor_seresnext_with_reduce_gpu
  test_parallel_executor_seresnext_with_fuse_all_reduce_gpu
  test_distributed_fused_lamb_op_with_clip
  test_distributed_fused_lamb_op_without_clip
  test_distributed_fused_lamb_op_with_gradient_merge
  PROPERTIES LABELS "RUN_TYPE=DIST")

if(NOT WIN32 AND NOT APPLE)
  set_tests_properties(test_imperative_signal_handler
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_imperative_data_loader_base
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_imperative_data_loader_fds_clear
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_static
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_dynamic
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_exception
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_iterable_dataset_static
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_iterable_dataset_dynamic
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_dataset
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_multiprocess_dataloader_static PROPERTIES TIMEOUT
                                                                      120)
endif()

if(NOT WIN32)
  set_tests_properties(test_multiprocess_reader_exception
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_layers PROPERTIES TIMEOUT 120)
  if(WITH_NV_JETSON)
    set_tests_properties(test_ir_memory_optimize_transformer PROPERTIES TIMEOUT
                                                                        1200)
  else()
    set_tests_properties(test_ir_memory_optimize_transformer PROPERTIES TIMEOUT
                                                                        120)
  endif()
endif()

if(WITH_DISTRIBUTE)
  set_tests_properties(test_dist_fleet_ctr2 PROPERTIES TIMEOUT 200)
  set_tests_properties(test_dist_fleet_sparse_embedding_ctr PROPERTIES TIMEOUT
                                                                       200)
  set_tests_properties(test_dist_fleet_infer PROPERTIES TIMEOUT 200)
  set_tests_properties(test_dist_fleet_raw_program_optimizer_fuse_allreduce
                       PROPERTIES TIMEOUT 60)
  set_tests_properties(test_dist_dygraph_apis PROPERTIES TIMEOUT 120)

  # NODE(Ruibiao): Remove it after static build is enabled by default.
  set_tests_properties(
    test_dist_mnist_fp16_allreduce test_dist_mnist_pg
    PROPERTIES ENVIRONMENT FLAGS_new_executor_static_build=true)
endif()

# setting timeout value as 15S
set_tests_properties(test_run PROPERTIES TIMEOUT 120)
set_tests_properties(test_sync_batch_norm_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_cross_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_lod_tensor_to_selected_rows
                     PROPERTIES TIMEOUT 200)
set_tests_properties(test_lstm_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_star_gan_with_gradient_penalty
                     PROPERTIES TIMEOUT 120)

set_tests_properties(test_bicubic_interp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_deformable_conv_op PROPERTIES TIMEOUT 200)
set_tests_properties(test_nearest_interp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_profiler PROPERTIES TIMEOUT 120)
set_tests_properties(test_inplace_softmax_with_cross_entropy PROPERTIES TIMEOUT
                                                                        120)
set_tests_properties(test_cross_entropy2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_cross_entropy_loss PROPERTIES TIMEOUT 180)
set_tests_properties(test_gru_unit_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_activation_nn_grad PROPERTIES TIMEOUT 200)
set_tests_properties(test_empty_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_parallel_executor_transformer PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_div_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_regularizer_api PROPERTIES TIMEOUT 150)
set_tests_properties(test_multiclass_nms_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_add_reader_dependency PROPERTIES TIMEOUT 120)
set_tests_properties(test_bilateral_slice_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_buffer_shared_memory_reuse_pass PROPERTIES TIMEOUT
                                                                     120)
set_tests_properties(test_fuse_relu_depthwise_conv_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_fleet_util PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_transformer_sorted_gradient
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_matmul_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_nearest_interp_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_trilinear_interp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_bicubic_interp_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_gather_op PROPERTIES TIMEOUT 200)
set_tests_properties(test_static_save_load PROPERTIES TIMEOUT 250)
set_tests_properties(test_pylayer_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_paddle_save_load_binary PROPERTIES TIMEOUT 120)
if(WIN32)
  set_tests_properties(test_static_save_load_large PROPERTIES TIMEOUT 900)
  set_tests_properties(test_paddle_save_load PROPERTIES TIMEOUT 250)
else()
  set_tests_properties(test_static_save_load_large PROPERTIES TIMEOUT 600)
  set_tests_properties(test_paddle_save_load PROPERTIES TIMEOUT 250)
endif()
if(WITH_NV_JETSON)
  set_tests_properties(test_concat_op PROPERTIES TIMEOUT 1200)
  set_tests_properties(test_conv3d_transpose_part2_op PROPERTIES TIMEOUT 1200)
  set_tests_properties(test_conv3d_transpose_op PROPERTIES TIMEOUT 1200)
  set_tests_properties(test_conv3d_op PROPERTIES TIMEOUT 1200)
  set_tests_properties(test_norm_op PROPERTIES TIMEOUT 1200)
  set_tests_properties(test_layer_norm_op PROPERTIES TIMEOUT 1500)
  set_tests_properties(test_pool3d_op PROPERTIES TIMEOUT 1500)
else()
  set_tests_properties(test_concat_op PROPERTIES TIMEOUT 120)
  set_tests_properties(test_conv3d_transpose_part2_op PROPERTIES TIMEOUT 120)
  set_tests_properties(test_conv3d_transpose_op PROPERTIES TIMEOUT 120)
  set_tests_properties(test_conv3d_op PROPERTIES TIMEOUT 120)
  set_tests_properties(test_norm_op PROPERTIES TIMEOUT 120)
  set_tests_properties(test_layer_norm_op PROPERTIES TIMEOUT 200)
  set_tests_properties(test_pool3d_op PROPERTIES TIMEOUT 150)
endif()
set_tests_properties(test_imperative_selected_rows_to_lod_tensor
                     PROPERTIES TIMEOUT 200)
set_tests_properties(test_index_select_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_index_add_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_tensordot PROPERTIES TIMEOUT 200)
set_tests_properties(test_partial_eager_deletion_transformer PROPERTIES TIMEOUT
                                                                        120)
set_tests_properties(test_parallel_executor_seresnext_with_reduce_gpu
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_dropout_op PROPERTIES TIMEOUT 200)
set_tests_properties(test_argsort_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_gather_nd_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_nn_grad PROPERTIES TIMEOUT 180)
set_tests_properties(test_elementwise_sub_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_row_conv_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_parallel_executor_seresnext_with_fuse_all_reduce_gpu
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_distributed_fused_lamb_op_with_clip PROPERTIES TIMEOUT
                                                                         120)
set_tests_properties(test_distributed_fused_lamb_op_without_clip
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_distributed_fused_lamb_op_with_gradient_merge
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_min_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_nan_inf PROPERTIES TIMEOUT 120)
set_tests_properties(test_deformable_conv_v1_op PROPERTIES TIMEOUT 300)
set_tests_properties(test_parallel_executor_transformer_auto_growth
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_add_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_weight_decay PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_ptb_rnn_sorted_gradient PROPERTIES TIMEOUT
                                                                        120)
set_tests_properties(test_crop_tensor_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_ptb_rnn PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_save_load_v2 PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_transpose_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_multiprocess_dataloader_iterable_dataset_static
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_lstm_cudnn_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_stack_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_bilinear_interp_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_svd_op PROPERTIES TIMEOUT 80)
set_tests_properties(test_einsum_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_qr_op PROPERTIES TIMEOUT 60)
set_tests_properties(test_trilinear_interp_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_masked_select_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_sigmoid_cross_entropy_with_logits_op
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_optimizer_v2 PROPERTIES TIMEOUT 150)
set_tests_properties(test_partial_sum_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_cond PROPERTIES TIMEOUT 120)
set_tests_properties(test_space_to_depth_op PROPERTIES TIMEOUT 200)
set_tests_properties(test_sgd_op PROPERTIES TIMEOUT 250)
set_tests_properties(test_parallel_executor_seresnext_base_gpu
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_norm_nn_grad PROPERTIES TIMEOUT 180)
set_tests_properties(test_matrix_nms_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_generator_dataloader PROPERTIES TIMEOUT 120)
set_tests_properties(test_partial_concat_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_fuse_optimizer_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_softmax_with_cross_entropy_op PROPERTIES TIMEOUT 220)
set_tests_properties(test_reduce_op PROPERTIES TIMEOUT 500)
set_tests_properties(test_adam_optimizer_fp32_fp64 PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_nn_grad PROPERTIES TIMEOUT 120)
set_tests_properties(
  test_buffer_shared_memory_reuse_pass_and_fuse_optimization_op_pass
  PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv_nn_grad PROPERTIES TIMEOUT 120)
set_tests_properties(test_program_prune_backward PROPERTIES TIMEOUT 120)
set_tests_properties(test_group_norm_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_optimizer PROPERTIES TIMEOUT 250)
set_tests_properties(test_imperative_optimizer_v2 PROPERTIES TIMEOUT 250)
set_tests_properties(test_pool2d_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_transpose_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_activation_op PROPERTIES TIMEOUT 270)
set_tests_properties(test_normal PROPERTIES TIMEOUT 120)
set_tests_properties(test_lstmp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_bilinear_interp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_decoupled_py_reader PROPERTIES TIMEOUT 120)
set_tests_properties(test_fuse_bn_act_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_op_depthwise_conv PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_api PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_mul_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_cyclic_cifar_dataset PROPERTIES TIMEOUT 120)
set_tests_properties(test_fuse_all_reduce_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_dygraph_multi_forward PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_ocr_attention_model PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_mnist PROPERTIES TIMEOUT 120)
set_tests_properties(test_fused_elemwise_activation_op PROPERTIES TIMEOUT 270)
set_tests_properties(test_fused_elemwise_activation_op
                     PROPERTIES LABELS "RUN_TYPE=NIGHTLY")
set_tests_properties(test_gru_op PROPERTIES TIMEOUT 200)
set_tests_properties(test_regularizer PROPERTIES TIMEOUT 150)
set_tests_properties(test_imperative_resnet PROPERTIES TIMEOUT 200)
set_tests_properties(test_imperative_resnet_sorted_gradient PROPERTIES TIMEOUT
                                                                       200)
set_tests_properties(test_imperative_se_resnext PROPERTIES TIMEOUT 200)
set_tests_properties(test_matmul_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_slice_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_strided_slice_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_translated_layer PROPERTIES TIMEOUT 120)
set_tests_properties(test_pad3d_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_dataloader_keep_order PROPERTIES TIMEOUT 120)
set_tests_properties(test_mean_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_dataloader_unkeep_order PROPERTIES TIMEOUT 120)
set_tests_properties(test_reader_reset PROPERTIES TIMEOUT 120)
set_tests_properties(test_pool3d_api PROPERTIES TIMEOUT 120)
set_tests_properties(test_cumprod_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_split_program PROPERTIES TIMEOUT 120)
set_tests_properties(test_graph_send_ue_recv_op PROPERTIES TIMEOUT 60)
set_tests_properties(test_graph_send_uv_op PROPERTIES TIMEOUT 60)

if(WITH_DISTRIBUTE
   AND WITH_GPU
   AND WITH_NCCL)
  set_tests_properties(test_auto_parallel_data_unshard PROPERTIES TIMEOUT 120)
  set_tests_properties(test_auto_parallel_save_load PROPERTIES TIMEOUT 120)
  set_tests_properties(test_auto_parallel_autoconvert PROPERTIES TIMEOUT 120)
endif()

if(APPLE)
  set_tests_properties(test_imperative_transformer_sorted_gradient
                       PROPERTIES TIMEOUT 300)
  set_tests_properties(test_multiclass_nms_op PROPERTIES TIMEOUT 300)
  set_tests_properties(test_weight_decay PROPERTIES TIMEOUT 300)
endif()

if((WITH_ROCM OR WITH_GPU) AND NOT WIN32)
  if(WITH_DISTRIBUTE)
    set_tests_properties(test_static_model_parallel_fused_feedforward
                         PROPERTIES TIMEOUT 120)
    set_tests_properties(test_static_model_parallel_fused_attention
                         PROPERTIES TIMEOUT 120)
    set_tests_properties(test_static_model_parallel_fused_multi_transformer
                         PROPERTIES TIMEOUT 120)
    set_tests_properties(test_pipeline_parallel PROPERTIES LABELS
                                                           "RUN_TYPE=DIST")
    set_tests_properties(test_reducescatter PROPERTIES TIMEOUT 120)
    set_tests_properties(test_allgather PROPERTIES TIMEOUT 120)
  endif()
  set_tests_properties(test_paddle_multiprocessing PROPERTIES TIMEOUT 120)
  set_tests_properties(test_pipeline_parallel PROPERTIES TIMEOUT 120)
endif()
if(WITH_GPU OR WITH_ROCM)
  set_tests_properties(test_rank_attention_op PROPERTIES TIMEOUT 120)
endif()
if(WITH_GPU AND NOT WIN32)
  set_tests_properties(test_fused_multi_transformer_int8_op PROPERTIES TIMEOUT
                                                                       60)
endif()
set_tests_properties(test_inplace_addto_strategy PROPERTIES TIMEOUT 120)
set_tests_properties(test_eigvals_op PROPERTIES TIMEOUT 400)
set_tests_properties(
  test_cuda_memory_reserved PROPERTIES ENVIRONMENT
                                       "FLAGS_allocator_strategy=auto_growth")
if(WITH_GLOO)
  set_tests_properties(test_parallel_dygraph_dataparallel_cpuonly
                       PROPERTIES TIMEOUT 30)
  set_tests_properties(test_parallel_dygraph_unused_variables_gloo
                       PROPERTIES TIMEOUT 120)
  set_tests_properties(test_parallel_dygraph_sparse_embedding_gloo
                       PROPERTIES TIMEOUT 120)
  set_tests_properties(test_parallel_dygraph_sparse_embedding_over_height_gloo
                       PROPERTIES TIMEOUT 120)
endif()

set(TEST_CINN_OPS
    test_softmax_op
    test_expand_v2_op
    test_reduce_op
    test_slice_op
    test_stack_op
    test_activation_op
    test_full_like_op
    test_index_select_op
    test_fill_any_like_op
    test_concat_op
    test_top_k_v2_op
    test_elementwise_add_op
    test_elementwise_sub_op
    test_elementwise_div_op
    test_elementwise_mul_op
    test_gather_nd_op
    test_squeeze2_op
    test_elementwise_pow_op
    test_elementwise_max_op
    test_transpose_op
    test_reshape_op
    test_mean_op
    test_unsqueeze2_op
    test_meshgrid_op
    test_scatter_op
    test_gather_op
    test_layer_norm_op
    test_cast_op
    test_dropout_op
    test_group_norm_op
    test_roll_op)

foreach(TEST_CINN_OPS ${TEST_CINN_OPS})
  if(WITH_CINN)
    set_tests_properties(${TEST_CINN_OPS} PROPERTIES LABELS "RUN_TYPE=CINN")
    set_tests_properties(${TEST_CINN_OPS} PROPERTIES TIMEOUT 150)
  endif()
endforeach()

if(WITH_CINN AND WITH_TESTING)
  set_tests_properties(
    test_resnet50_with_cinn
    PROPERTIES
      LABELS
      "RUN_TYPE=CINN"
      ENVIRONMENT
      FLAGS_allow_cinn_ops="conv2d;conv2d_grad;elementwise_add;elementwise_add_grad;relu;relu_grad;sum"
  )
  set_tests_properties(
    test_parallel_executor_run_cinn
    PROPERTIES
      LABELS
      "RUN_TYPE=CINN"
      ENVIRONMENT
      FLAGS_allow_cinn_ops="conv2d;conv2d_grad;elementwise_add;elementwise_add_grad;relu;relu_grad;sum"
  )
endif()

set_tests_properties(
  test_cuda_graph_static_mode
  PROPERTIES ENVIRONMENT "FLAGS_CUDA_GRAPH_USE_STANDALONE_EXECUTOR=1")
set_tests_properties(
  test_cuda_graph_static_mode_error
  PROPERTIES ENVIRONMENT "FLAGS_CUDA_GRAPH_USE_STANDALONE_EXECUTOR=1")

# These UTs are to temporarily test static build for standalone_executor, will be removed after static build is enabled by default.
set(STATIC_BUILD_TESTS
    test_adagrad_op
    test_adamw_op
    test_arg_min_max_op
    test_bincount_op
    test_decoupled_py_reader
    test_fake_quantize_op
    test_fetch_lod_tensor_array
    test_imperative_optimizer
    test_lamb_op
    test_layer_norm_op
    test_lookup_table_bf16_op
    test_lookup_table_v2_op
    test_matmul_op
    test_matmul_v2_op
    test_merged_adam_op
    test_momentum_op
    test_nce
    test_paddle_save_load_binary
    test_reduce_op
    test_segment_ops
    test_sparse_momentum_op
    test_sgd_op_bf16
    test_softmax_mask_fuse_upper_triangle_op
    test_sparse_conv_op
    test_sparse_norm_op
    test_sparse_pooling_op
    test_tensor_array_to_tensor
    test_while_op
    test_one_hot_v2_op)

foreach(STATIC_BUILD_TEST ${STATIC_BUILD_TESTS})
  py_test_modules(
    ${STATIC_BUILD_TEST}_static_build MODULES ${STATIC_BUILD_TEST} ENVS
    FLAGS_new_executor_static_build=true)
endforeach()

set_tests_properties(test_decoupled_py_reader_static_build PROPERTIES TIMEOUT
                                                                      120)
set_tests_properties(test_imperative_optimizer_static_build PROPERTIES TIMEOUT
                                                                       250)
set_tests_properties(test_matmul_op_static_build PROPERTIES TIMEOUT 120)
set_tests_properties(test_matmul_v2_op_static_build PROPERTIES TIMEOUT 120)
set_tests_properties(test_layer_norm_op_static_build PROPERTIES TIMEOUT 1500)
set_tests_properties(test_paddle_save_load_binary_static_build
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_reduce_op_static_build PROPERTIES TIMEOUT 500)

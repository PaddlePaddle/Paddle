# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


function collect_failed_tests() {
    for file in `ls $tmp_dir`; do
        exit_code=0
        grep -q 'The following tests FAILED:' $tmp_dir/$file||exit_code=$?
        if [ $exit_code -ne 0 ]; then
            failuretest=''
        else
            failuretest=`grep -A 10000 'The following tests FAILED:' $tmp_dir/$file | sed 's/The following tests FAILED://g'|sed '/^$/d'`
            failed_test_lists="${failed_test_lists}
            ${failuretest}"
        fi
    done
}

serial_list="^test_conv2d_op$|\
^test_conv2d_transpose_op$|\
^test_conv3d_op$"

parallel_list="^init_phi_test$|\
^operator_test$|\
^test_collective_cpu_barrier_with_gloo$|\
^test_conv1d_layer$|\
^test_conv1d_transpose_layer$|\
^test_conv2d_api$|\
^test_conv2d_fusion_op$|\
^test_conv2d_layer$|\
^test_conv2d_op_depthwise_conv$|\
^test_conv2d_transpose_layer$|\
^test_conv2d_transpose_op_depthwise_conv$|\
^test_conv3d_layer$|\
^test_conv3d_transpose_layer$|\
^test_conv3d_transpose_op$|\
^test_conv_bn_fuse_pass_cc$|\
^test_conv_nn_grad$|\
^test_conv_shift_op$|\
^test_conv_transpose_nn_grad$|\
^test_convert_call$|\
^test_convert_call_generator$|\
^test_convert_operators$|\
^test_convert_to_mixed_precision$|\
^test_convert_to_process_meshes$|\
^test_cpu_cuda_to_tensor$|\
^test_cudnn_placement_pass$|\
^test_custom_kernel$|\
^test_dist_fleet_ps11$|\
^test_dist_fleet_ps12$|\
^test_dygraph_sharding_stage2_bf16$|\
^test_executor_feed_non_tensor$|\
^test_flash_attention$|\
^test_fused_adam_op$|\
^test_fused_attention_no_dropout$|\
^test_fused_attention_op$|\
^test_fused_attention_op_api$|\
^test_fused_attention_op_api_static_build$|\
^test_fused_attention_op_static_build$|\
^test_fused_bias_dropout_residual_layer_norm_op$|\
^test_fused_bias_dropout_residual_layer_norm_op_api$|\
^test_fused_comm_buffer$|\
^test_fused_dropout_act_bias$|\
^test_fused_dropout_add_op$|\
^test_fused_emb_seq_pool_op$|\
^test_fused_embedding_fc_lstm_op$|\
^test_fused_fc_elementwise_layernorm_op$|\
^test_fused_feedforward_op$|\
^test_fused_feedforward_op_static_build$|\
^test_fused_gate_attention_op$|\
^test_fused_gemm_epilogue_grad_op$|\
^test_fused_gemm_epilogue_grad_op_with_es$|\
^test_fused_gemm_epilogue_op$|\
^test_fused_gemm_epilogue_op_with_es$|\
^test_fused_layernorm_residual_dropout_bias$|\
^test_fused_linear_param_grad_add$|\
^test_fused_linear_pass$|\
^test_fused_matmul_bias$|\
^test_fused_multi_transformer_decoder_pass$|\
^test_fused_multi_transformer_encoder_pass$|\
^test_fused_multi_transformer_int8_op$|\
^test_fused_residual_dropout_bias$|\
^test_fused_rotary_position_embedding$|\
^test_fused_scale_bias_relu_conv_bnstats_op$|\
^test_fused_token_prune_op$|\
^test_fused_transformer_encoder_layer$|\
^test_fused_transformer_with_amp_decorator$|\
^test_gather_nd_op$|\
^test_index_select_op$|\
^test_pass_base_list$|\
^test_roll_op$|\
^test_switch_autotune$|\
^test_tcp_store$|\
^test_to_tensor$|\
^test_top_k_v2_op$"

cd ${work_dir}/build
tmp_dir=`mktemp -d`
tmpfile_rand=`date +%s%N`
tmpfile=$tmp_dir/$tmpfile_rand"_"$i
set +e
ctest --output-on-failure -R "($parallel_list)" --timeout 120 -j4 | tee -a $tmpfile; test ${PIPESTATUS[0]} -eq 0;
EXIT_CODE_1=$?

ctest --output-on-failure -R "($serial_list)" --timeout 120 -j1 | tee -a $tmpfile; test ${PIPESTATUS[0]} -eq 0;
EXIT_CODE_2=$?
set -e

if [ "${EXIT_CODE_1}" != "0" ] || [ "${EXIT_CODE_2}" != "0" ];then
  echo "Sorry, some tests failed."
  collect_failed_tests
  rm -f $tmp_dir/*
  echo "Summary Failed Tests... "
  echo "========================================"
  echo "The following tests FAILED: "
  echo "${failuretest}" | sort -u
  exit 8
fi

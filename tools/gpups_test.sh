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

failuretest=''
function collect_failed_tests() {
    for file in `ls $tmp_dir`; do
        exit_code=0
        grep -q 'The following tests FAILED:' $tmp_dir/$file||exit_code=$?
        if [ $exit_code -eq 0 ]; then
            failuretest=`grep -A 10000 'The following tests FAILED:' $tmp_dir/$file | sed 's/The following tests FAILED://g'|sed '/^$/d'|grep -v 'Passed'`
	    failed_test_lists="${failuretest}
            ${failed_test_lists}"
        fi
    done
}

function get_quickly_disable_ut() {
    python -m pip install httpx
    if disable_ut_quickly=$(python ${PADDLE_ROOT}/tools/get_quick_disable_lt.py); then
        echo "========================================="
        echo "The following unittests have been disabled:"
        echo ${disable_ut_quickly}
        echo "========================================="
    else

        exit 102
        disable_ut_quickly='disable_ut'
    fi
}

# disable test:
# test_dygraph_dataparallel_bf16
# test_dygraph_sharding_stage2_bf16
# test_dygraph_sharding_stage3_bf16
# test_dygraph_sharding_stage1_fp16
# test_dygraph_sharding_stage1_bf16

serial_list="^test_conv2d_op$|\
^test_conv2d_transpose_op$|\
^test_dist_fuse_resunit_pass$|\
^test_conv3d_op$"

parallel_list="^init_phi_test$|\
^operator_test$|\
^test_tcp_store$|\
^test_collective_cpu_barrier_with_gloo$|\
^test_conv1d_layer$|\
^test_conv1d_transpose_layer$|\
^test_conv2d_api_deprecated$|\
^test_conv2d_layer_deprecated$|\
^test_conv2d_op_depthwise_conv$|\
^test_conv2d_transpose_layer$|\
^test_conv2d_transpose_op_depthwise_conv$|\
^test_conv3d_layer$|\
^test_conv3d_transpose_layer$|\
^test_conv3d_transpose_op$|\
^test_conv_bn_fuse_pass_cc$|\
^test_conv_nn_grad$|\
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
^test_executor_feed_non_tensor$|\
^test_fuse_resunit_pass$|\
^test_fused_adam_op$|\
^test_fused_attention_no_dropout$|\
^test_fused_attention_op$|\
^test_fused_attention_op_api$|\
^test_fused_attention_op_api_static_build$|\
^test_fused_attention_op_static_build$|\
^test_fused_dconv_drelu_dbn_op$|\
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
^test_fused_gemm_epilogue_grad_op$|\
^test_fused_gemm_epilogue_grad_op_with_es$|\
^test_fused_gemm_epilogue_op$|\
^test_fused_gemm_epilogue_op_with_es$|\
^test_fused_layernorm_residual_dropout_bias$|\
^test_fused_linear_param_grad_add$|\
^test_fused_linear_pass_deprecated$|\
^test_fused_matmul_bias$|\
^test_fused_multi_transformer_decoder_pass$|\
^test_fused_multi_transformer_encoder_pass$|\
^test_fused_residual_dropout_bias$|\
^test_fused_rotary_position_embedding$|\
^test_fused_scale_bias_add_relu_op$|\
^test_fused_scale_bias_relu_conv_bn_op$|\
^test_fused_token_prune_op$|\
^test_fused_transformer_encoder_layer$|\
^test_fused_transformer_with_amp_decorator$|\
^test_fused_dot_product_attention_op$|\
^test_fused_dot_product_attention_op_static$|\
^test_fuse_dot_product_attention_pass$|\
^test_fused_dot_product_attention_pass$|\
^test_gather_nd_op$|\
^test_index_select_op$|\
^test_pass_base_list$|\
^test_pool_max_op$|\
^test_roll_op$|\
^test_switch_autotune$|\
^test_to_tensor$|\
^test_top_k_v2_op$|\
^test_pir_amp$"

cd ${work_dir}/build
tmp_dir=`mktemp -d`
tmpfile_rand=`date +%s%N`
tmpfile1_rand=`date +%s%N`
tmpfile=$tmp_dir/$tmpfile_rand"_"$i
tmpfile1=$tmp_dir/$tmpfile1_rand"_"$i
set +e

get_quickly_disable_ut||disable_ut_quickly='disable_ut'

NUM_PROC=4
EXIT_CODE=0
pids=()
for (( i = 0; i < $NUM_PROC; i++ )); do
    cuda_list="$((i*2)),$((i*2+1))"
    (env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC --output-on-failure -R "($parallel_list)" -E "($disable_ut_quickly)" --timeout 120 -j4 | tee -a $tmpfile; test ${PIPESTATUS[0]} -eq 0)&
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait $pid
    status=$?
    if [ $status -ne 0 ]; then
        EXIT_CODE=8
    fi
done

pids=()
for (( i = 0; i < $NUM_PROC; i++ )); do
    cuda_list="$((i*2)),$((i*2+1))"
    (env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC --output-on-failure -R "($serial_list)" -E "($disable_ut_quickly)" --timeout 180 -j1 | tee -a $tmpfile1; test ${PIPESTATUS[0]} -eq 0)&
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait $pid
    status=$?
    if [ $status -ne 0 ]; then
        EXIT_CODE=8
    fi
done

set -e

if [ "${EXIT_CODE}" != "0" ];then
  echo "Sorry, some tests failed."
  collect_failed_tests
  echo "Summary Failed Tests... "
  echo "========================================"
  echo "The following tests FAILED: "
  echo "${failed_test_lists}"| sort -u
  exit 8
fi

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

serial_list="^test_parallel_dygraph_control_flow$|\
^test_eager_dist_api$|\
^test_collective_process_group$|\
^test_parallel_dygraph_sparse_embedding$|\
^test_parallel_dygraph_unused_variables$|\
^test_static_model_parallel$|\
^test_dygraph_sharding_stage2$|\
^test_parallel_dygraph_no_sync$|\
^test_parallel_dygraph_mnist$|\
^test_dygraph_group_sharded_api_for_eager$|\
^test_new_api_per_op_and_group_intranode$"

concurrency_list="^test_fp8_deep_gemm$|\
^test_fp8_quant$|\
^test_fused_act_dequant_op$|\
^test_fused_stack_transpose_quant_op$|\
^test_fused_swiglu_weighted_bwd_op$|\
^test_fused_transpose_wlch_split_quant_op$|\
^test_fused_weighted_swiglu_act_quant_op$|\
^test_incubate_build_src_rank_and_local_expert_id$|\
^test_incubate_cal_aux_loss$|\
^test_incubate_cross_entropy_with_softmax_bwd_w_downcast$|\
^test_incubate_embedding_grad$|\
^test_incubate_expand_modality_expert_id$|\
^test_incubate_fused_loss$|\
^test_incubate_fused_rmsnorm_ext$|\
^test_incubate_int_bincount$|\
^test_incubate_moe_combine$|\
^test_incubate_moe_combine_no_weight$|\
^test_incubate_moe_gate_dispatch_and_quant$|\
^test_incubate_moe_gate_dispatch_partial_nosoftmaxtopk$|\
^test_incubate_moe_gate_dispatch_w_permute$|\
^test_incubate_moe_gate_dispatch_w_permute_bwd$|\
^test_moe_permute_unpermute$|\
^test_buffer_shared_memory_reuse_pass$|\
^test_buffer_shared_memory_reuse_pass_and_fuse_optimization_op_pass$|\
^test_fleet_perf_test$|\
^test_pipeline_parallel$|\
^test_install_check$|\
^test_install_check_pir$|\
^test_c_concat$|\
^test_c_split$|\
^test_collective_allgather_api$|\
^test_collective_allgather_object_api$|\
^test_collective_allreduce_api$|\
^test_collective_alltoall_api$|\
^test_collective_alltoall_single$|\
^test_collective_alltoall_single_api$|\
^test_collective_barrier_api$|\
^test_collective_batch_isend_irecv$|\
^test_collective_broadcast_api$|\
^test_collective_broadcast_object_list_api$|\
^test_collective_concat_api$|\
^test_collective_cpu_barrier_with_gloo$|\
^test_collective_global_gather$|\
^test_collective_global_scatter$|\
^test_collective_isend_irecv_api$|\
^test_collective_optimizer$|\
^test_collective_process_group_pir$|\
^test_collective_reduce_api$|\
^test_collective_reduce_scatter$|\
^test_collective_reduce_scatter_api$|\
^test_collective_scatter_api$|\
^test_collective_scatter_object_list_api$|\
^test_collective_gather_api$|\
^test_collective_sendrecv_api$|\
^test_collective_split_embedding_none_divisible$|\
^test_communication_stream_allgather_api$|\
^test_communication_stream_allreduce_api$|\
^test_communication_stream_alltoall_api$|\
^test_communication_stream_alltoall_single_api$|\
^test_communication_stream_broadcast_api$|\
^test_communication_stream_reduce_api$|\
^test_communication_stream_reduce_scatter_api$|\
^test_communication_stream_scatter_api$|\
^test_communication_stream_sendrecv_api$|\
^test_gen_nccl_id_op$|\
^test_new_group_api$|\
^test_world_size_and_rank$|\
^test_strategy_group$|\
^test_orthogonal_strategy$|\
^test_comm_group_num$|\
^test_parallel_margin_cross_entropy$|\
^test_parallel_dygraph_transformer$|\
^test_parallel_dygraph_mp_layers$|\
^test_tcp_store$|\
^test_dygraph_sharding_stage3_for_eager$|\
^test_dygraph_sharding_stage3_bf16$|\
^test_parallel_dygraph_pipeline_parallel$|\
^test_parallel_dygraph_pipeline_parallel_shared_weight$|\
^test_parallel_dygraph_pipeline_parallel_sync_send$|\
^test_parallel_dygraph_pipeline_parallel_with_virtual_stage$|\
^test_parallel_dygraph_pp_adaptor$|\
^test_parallel_class_center_sample$|\
^test_dygraph_dataparallel_bf16$|\
^test_dygraph_sharding_stage2_bf16$|\
^test_dygraph_sharding_stage1_fp16$|\
^test_parallel_dygraph_sharding_parallel$|\
^test_parallel_dygraph_tensor_parallel$|\
^test_parallel_dygraph_sep_parallel$|\
^test_parallel_dygraph_no_sync_gradient_check$|\
^test_parallel_dygraph_qat$|\
^test_parallel_dygraph_sparse_embedding_over_height$|\
^test_new_group$|\
^test_c_comm_init_op$|\
^test_parallel_dygraph_se_resnext$|\
^test_parallel_dygraph_sync_batch_norm$|\
^test_imperative_auto_mixed_precision_for_eager$|\
^test_dist_se_resnext_dgc$|\
^test_fleet_log$|\
^test_dygraph_dist_save_load$|\
^test_dualpipe$|\
^test_zero_bubble_utils$|\
^test_shutdown_process_group$|\
^test_pp_send_recv_dict$|\
^test_pp_unified_dygraph_model$|\
^test_sharding_stage3_bugfix$|\
^test_build_cinn_pass_resnet$|\
^test_dist_fuse_all_reduce_pass$|\
^test_dist_fuse_gemm_epilogue_pass$|\
^test_fuse_allreduce_split_to_reducescatter_pass$|\
^test_ps_server_pass$|\
^test_white_lists$"

cd ${work_dir}/build
tmp_dir=`mktemp -d`
tmpfile_rand=`date +%s%N`
tmpfile1_rand=`date +%s%N`
tmpfile=$tmp_dir/$tmpfile_rand"_"$i
tmpfile1=$tmp_dir/$tmpfile1_rand"_"$i
set +e

get_quickly_disable_ut||disable_ut_quickly='disable_ut'
disable_ut_quickly="$disable_ut_quickly|\
^test_parallel_dygraph_sparse_embedding$|\
^test_parallel_dygraph_unused_variables$|\
^test_static_model_parallel$|\
^test_parallel_dygraph_sync_batch_norm$|\
^test_parallel_dygraph_no_sync$|\
^test_parallel_dygraph_control_flow$|\
^test_parallel_dygraph_no_sync$|\
^test_orthogonal_strategy$|\
^test_collective_alltoall_single$|\
^test_collective_process_group$|\
^test_parallel_dygraph_transformer$|\
^test_new_api_per_op_and_group_intranode$|\
^test_communication_stream_reduce_api$"

NUM_PROC=4
EXIT_CODE=0
pids=()
for (( i = 0; i < $NUM_PROC; i++ )); do
    (ctest -I $i,,$NUM_PROC --output-on-failure -R "($concurrency_list)" -E "($disable_ut_quickly)" --timeout 120 -j1 | tee -a $tmpfile; test ${PIPESTATUS[0]} -eq 0)&
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait $pid
    status=$?
    if [ $status -ne 0 ]; then
        EXIT_CODE=8
    fi
done

NUM_PROC=1
pids=()
for (( i = 0; i < $NUM_PROC; i++ )); do
    (ctest -I $i,,$NUM_PROC --output-on-failure -R "($serial_list)" -E "($disable_ut_quickly)" --timeout 120 -j1 | tee -a $tmpfile1; test ${PIPESTATUS[0]} -eq 0)&
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

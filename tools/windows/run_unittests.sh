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


# /*================Fixed Disabled Windows CUDA10.x MKL(PR-CI-Windows) unittests===========================*/
# TODO: fix these unittest that is bound to fail
disable_wingpu_test="^test_model$|\
^test_dataloader_early_reset$|\
^test_add_reader_dependency$|\
^test_add_reader_dependency_for_interpretercore$|\
^test_decoupled_py_reader$|\
^test_generator_dataloader$|\
^test_parallel_dygraph_sync_batch_norm$|\
^test_py_reader_using_executor$|\
^test_parallel_executor_seresnext_base_gpu$|\
^test_parallel_executor_seresnext_with_fuse_all_reduce_gpu$|\
^test_parallel_executor_seresnext_with_reduce_gpu$|\
^test_program_prune_backward$|\
^test_decoupled_py_reader_data_check$|\
^test_fleet_base_single$|\
^test_multiprocess_dataloader_iterable_dataset_dynamic$|\
^test_parallel_executor_feed_persistable_var$|\
^test_parallel_executor_inference_feed_partial_data$|\
^test_parallel_ssa_graph_inference_feed_partial_data$|\
^test_py_reader_combination$|\
^test_py_reader_pin_memory$|\
^test_py_reader_push_pop$|\
^test_reader_reset$|\
^test_imperative_se_resnext$|\
^test_sync_batch_norm_op$|\
^test_imperative_static_runner_while$|\
^test_dataloader_keep_order$|\
^test_dataloader_unkeep_order$|\
^test_multiprocess_dataloader_iterable_dataset_static$|\
^test_fuse_bn_act_pass$|\
^test_fuse_bn_add_act_pass$|\
^test_gather_op$|\
^test_activation_op$|\
^test_norm_nn_grad$|\
^test_bilinear_interp_op$|\
^disable_wingpu_test$"

# /*=================Fixed Disabled Windows TRT MKL unittests=======================*/
# TODO: fix these unittest that is bound to fail
disable_win_trt_test="^test_trt_convert_conv2d$|\
^test_trt_convert_conv2d_fusion$|\
^test_trt_convert_conv2d_transpose$|\
^test_trt_convert_depthwise_conv2d$|\
^test_trt_convert_emb_eltwise_layernorm$|\
^test_trt_convert_pool2d$|\
^test_trt_conv3d_op$|\
^test_trt_subgraph_pass$|\
^test_trt_convert_dropout$|\
^test_trt_convert_hard_sigmoid$|\
^test_trt_convert_reduce_mean$|\
^test_trt_convert_reduce_sum$|\
^test_trt_convert_group_norm$|\
^test_trt_convert_batch_norm$|\
^test_trt_convert_activation$|\
^test_trt_convert_depthwise_conv2d_transpose$|\
^test_trt_convert_elementwise$|\
^test_trt_convert_matmul$|\
^test_trt_convert_scale$"

# /*==========Fixed Disabled Windows CUDA11.x inference_api_test(PR-CI-Windows-Inference) unittests=============*/
disable_win_inference_test="^trt_quant_int8_yolov3_r50_test$|\
^test_trt_dynamic_shape_ernie$|\
^test_trt_dynamic_shape_ernie_fp16_ser_deser$|\
^lite_resnet50_test$|\
^test_trt_dynamic_shape_transformer_prune$|\
^lite_mul_model_test$|\
^trt_split_converter_test$|\
^paddle_infer_api_copy_tensor_tester$|\
^test_trt_deformable_conv$|\
^test_imperative_triple_grad$|\
^test_full_name_usage$|\
^test_trt_convert_unary$|\
^test_eigh_op$|\
^test_fc_op$|\
^test_stack_op$|\
^trt_split_converter_test$|\
^paddle_infer_api_copy_tensor_tester$|\
^test_var_base$|\
^test_einsum_v2$|\
^test_tensor_scalar_type_promotion_static$|\
^test_matrix_power_op$|\
^test_deformable_conv_v1_op$|\
^test_where_index$|\
^test_custom_grad_input$|\
^test_conv3d_transpose_op$|\
^test_conv_elementwise_add_act_fuse_pass$|\
^test_conv_eltwiseadd_bn_fuse_pass$|\
^test_custom_relu_op_setup$|\
^test_conv3d_transpose_part2_op$|\
^test_deform_conv2d$|\
^test_matmul_op$|\
^test_basic_api_transformation$|\
^test_deformable_conv_op$|\
^test_variable$|\
^test_mkldnn_conv_hard_sigmoid_fuse_pass$|\
^test_mkldnn_conv_hard_swish_fuse_pass$|\
^test_conv_act_mkldnn_fuse_pass$|\
^test_matmul_scale_fuse_pass$|\
^test_addmm_op$|\
^test_inverse_op$|\
^test_set_value_op$|\
^test_fused_multihead_matmul_op$|\
^test_cudnn_bn_add_relu$|\
^test_cond$|\
^test_conv_bn_fuse_pass$|\
^test_graph_khop_sampler$|\
^test_gru_rnn_op$|\
^test_masked_select_op$|\
^test_ir_fc_fuse_pass$|\
^test_fc_elementwise_layernorm_fuse_pass$|\
^test_linalg_pinv_op$|\
^test_math_op_patch_var_base$|\
^test_slice$|\
^test_conv_elementwise_add_fuse_pass$|\
^test_executor_and_mul$|\
^test_analyzer_int8_resnet50$|\
^test_analyzer_int8_mobilenetv1$|\
^test_trt_conv_pass$|\
^test_roll_op$|\
^test_lcm$|\
^test_elementwise_floordiv_op$|\
^test_autograd_functional_dynamic$|\
^test_corr$|\
^test_trt_convert_deformable_conv$|\
^test_conv_elementwise_add2_act_fuse_pass$|\
^test_tensor_scalar_type_promotion_dynamic$|\
^test_model$|\
^test_py_reader_combination$|\
^test_trt_convert_flatten$|\
^test_py_reader_push_pop$|\
^test_parallel_executor_feed_persistable_var$|\
^test_parallel_executor_inference_feed_partial_data$|\
^test_parallel_ssa_graph_inference_feed_partial_data$|\
^test_reader_reset$|\
^test_parallel_executor_seresnext_base_gpu$|\
^test_py_reader_pin_memory$|\
^test_multiprocess_dataloader_iterable_dataset_dynamic$|\
^test_multiprocess_dataloader_iterable_dataset_static$|\
^test_add_reader_dependency$|\
^test_add_reader_dependency_for_interpretercore$|\
^test_compat$|\
^test_decoupled_py_reader$|\
^test_generator_dataloader$|\
^test_py_reader_using_executor$|\
^test_imperative_static_runner_while$|\
^test_dataloader_keep_order$|\
^test_dataloader_unkeep_order$|\
^test_sync_batch_norm_op$|\
^test_fuse_bn_act_pass$|\
^test_fuse_bn_add_act_pass$|\
^test_decoupled_py_reader_data_check$|\
^test_parallel_dygraph_sync_batch_norm$|\
^test_dataloader_early_reset$|\
^test_fleet_base_single$|\
^test_sequence_pool$|\
^test_simplify_with_basic_ops_pass_autoscan$|\
^test_trt_activation_pass$|\
^test_trt_convert_hard_swish$|\
^test_trt_convert_leaky_relu$|\
^test_trt_convert_multihead_matmul$|\
^test_trt_convert_prelu$|\
^test_trt_fc_fuse_quant_dequant_pass$|\
^test_unsqueeze2_eltwise_fuse_pass$|\
^test_parallel_executor_seresnext_with_fuse_all_reduce_gpu$|\
^test_parallel_executor_seresnext_with_reduce_gpu$|\
^test_api_impl$|\
^test_tensordot$|\
^disable_win_inference_test$"


# /*==========Fixed Disabled Windows CPU OPENBLAS((PR-CI-Windows-OPENBLAS)) unittests==============================*/
# TODO: fix these unittest that is bound to fail
disable_wincpu_test="^jit_kernel_test$|\
^test_analyzer_transformer$|\
^test_vision_models$|\
^test_dygraph_multi_forward$|\
^test_imperative_transformer_sorted_gradient$|\
^test_program_prune_backward$|\
^test_imperative_resnet$|\
^test_imperative_resnet_sorted_gradient$|\
^test_imperative_se_resnext$|\
^test_imperative_static_runner_mnist$|\
^test_bmn$|\
^test_mobile_net$|\
^test_resnet_v2$|\
^test_build_strategy$|\
^test_se_resnet$|\
^disable_wincpu_test$"

# these unittest that cost long time, diabled temporarily, Maybe moved to the night
long_time_test="^test_gru_op$|\
^decorator_test$|\
^test_dataset_imdb$|\
^test_datasets$|\
^test_pretrained_model$|\
^test_gather_op$|\
^test_gather_nd_op$|\
^test_sequence_conv$|\
^test_space_to_depth_op$|\
^test_activation_nn_grad$|\
^test_activation_op$|\
^test_bicubic_interp_v2_op$|\
^test_bilinear_interp_v2_op$|\
^test_crop_tensor_op$|\
^test_cross_entropy2_op$|\
^test_cross_op$|\
^test_elementwise_nn_grad$|\
^test_fused_elemwise_activation_op$|\
^test_imperative_lod_tensor_to_selected_rows$|\
^test_imperative_selected_rows_to_lod_tensor$|\
^test_layer_norm_op$|\
^test_multiclass_nms_op$|\
^test_nearest_interp_v2_op$|\
^test_nn_grad$|\
^test_norm_nn_grad$|\
^test_normal$|\
^test_pool3d_op$|\
^test_static_save_load$|\
^test_trilinear_interp_op$|\
^test_trilinear_interp_v2_op$|\
^test_bilinear_interp_op$|\
^test_nearest_interp_op$|\
^test_sequence_conv$|\
^test_sgd_op$|\
^test_transformer$|\
^test_imperative_auto_mixed_precision$|\
^test_trt_matmul_quant_dequant$|\
^test_strided_slice_op$"


# /*============================================================================*/

set -e
set +x
NIGHTLY_MODE=$1
PRECISION_TEST=$2
WITH_GPU=$3

# Step1: Print disable_ut_quickly
export PADDLE_ROOT="$(cd "$PWD/../" && pwd )"
if [ ${NIGHTLY_MODE:-OFF} == "ON" ]; then
    nightly_label=""
else
    nightly_label="(RUN_TYPE=NIGHTLY|RUN_TYPE=DIST:NIGHTLY|RUN_TYPE=EXCLUSIVE:NIGHTLY)"
    echo "========================================="
    echo "Unittests with nightly labels  are only run at night"
    echo "========================================="
fi

if disable_ut_quickly=$(python ${PADDLE_ROOT}/tools/get_quick_disable_lt.py); then
    echo "========================================="
    echo "The following unittests have been disabled:"
    echo ${disable_ut_quickly}
    echo "========================================="
else
    disable_ut_quickly=''
fi

# Step2: Check added ut
set +e
cp $PADDLE_ROOT/tools/check_added_ut.sh $PADDLE_ROOT/tools/check_added_ut_win.sh
bash $PADDLE_ROOT/tools/check_added_ut_win.sh
rm -rf $PADDLE_ROOT/tools/check_added_ut_win.sh
if [ -f "$PADDLE_ROOT/added_ut" ];then
    added_uts=^$(awk BEGIN{RS=EOF}'{gsub(/\n/,"$|^");print}' $PADDLE_ROOT/added_ut)$
    ctest -R "(${added_uts})" -E "${disable_win_inference_test}" --output-on-failure -C Release --repeat-until-fail 3;added_ut_error=$?
    rm -f $PADDLE_ROOT/added_ut
    if [ "$added_ut_error" != 0 ];then
        echo "========================================"
        echo "Added UT should pass three additional executions"
        echo "========================================"
        exit 8;
    fi
fi


# Step3: Get precision UT and intersect with parallel UT, generate tools/*_new file
set -e
if [ ${WITH_GPU:-OFF} == "ON" ];then
    export CUDA_VISIBLE_DEVICES=0

    ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d' > all_ut_list
    num=$(ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d' | wc -l)
    echo "Windows 1 card TestCases count is $num"
    if [ ${PRECISION_TEST:-OFF} == "ON" ]; then
        python ${PADDLE_ROOT}/tools/get_pr_ut.py || echo "Failed to obtain ut_list !"
    fi
    
    python ${PADDLE_ROOT}/tools/group_case_for_parallel.py ${PADDLE_ROOT}
    
fi

failed_test_lists=''
tmp_dir=`mktemp -d`
function collect_failed_tests() {
    set +e
    for file in `ls $tmp_dir`; do
        grep -q 'The following tests FAILED:' $tmp_dir/$file
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            failuretest=''
        else
            failuretest=`grep -A 10000 'The following tests FAILED:' $tmp_dir/$file | sed 's/The following tests FAILED://g'|sed '/^$/d'`
            failed_test_lists="${failed_test_lists}
            ${failuretest}"
        fi
    done
    set -e
}

function run_unittest_cpu() {
    tmpfile=$tmp_dir/$RANDOM
    (ctest -E "$disable_ut_quickly|$disable_wincpu_test" -LE "${nightly_label}" --output-on-failure -C Release -j 8 | tee $tmpfile) &
    wait;
}

function run_unittest_gpu() {
    test_case=$1
    parallel_job=$2
    parallel_level_base=${CTEST_PARALLEL_LEVEL:-1}
    if [ "$2" == "" ]; then
        parallel_job=$parallel_level_base
    else
        # set parallel_job according to CUDA memory and suggested parallel num,
        # the latter is derived in linux server with 16G CUDA memory.
        cuda_memory=$(nvidia-smi --query-gpu=memory.total --format=csv | tail -1 | awk -F ' ' '{print $1}')
        parallel_job=$(($2 * $cuda_memory / 16000))
        if [ $parallel_job -lt 1 ]; then
            parallel_job=1
        fi
    fi
    echo "************************************************************************"
    echo "********These unittests run $parallel_job job each time with 1 GPU**********"
    echo "************************************************************************"
    export CUDA_VISIBLE_DEVICES=0

    if nvcc --version | grep 11.2; then
        disable_wingpu_test=${disable_win_inference_test}
    fi

    tmpfile=$tmp_dir/$RANDOM
    (ctest -R "$test_case" -E "$disable_ut_quickly|$disable_wingpu_test|$disable_win_trt_test|$long_time_test" -LE "${nightly_label}" --output-on-failure -C Release -j $parallel_job | tee $tmpfile ) &
    wait;
}

function unittests_retry(){
    is_retry_execuate=0
    wintest_error=1
    retry_time=3
    exec_times=0
    exec_retry_threshold=30
    retry_unittests=$(echo "${failed_test_lists}" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
    need_retry_ut_counts=$(echo "$retry_unittests" |awk -F ' ' '{print }'| sed '/^$/d' | wc -l)
    retry_unittests_regular=$(echo "$retry_unittests" |awk -F ' ' '{print }' | awk 'BEGIN{ all_str=""}{if (all_str==""){all_str=$1}else{all_str=all_str"$|^"$1}} END{print "^"all_str"$"}')
    tmpfile=$tmp_dir/$RANDOM

    if [ $need_retry_ut_counts -lt $exec_retry_threshold ];then
            retry_unittests_record=''
            while ( [ $exec_times -lt $retry_time ] )
                do
                    retry_unittests_record="$retry_unittests_record$failed_test_lists"
                    if ( [[ "$exec_times" == "0" ]] );then
                        cur_order='first'
                    elif ( [[ "$exec_times" == "1" ]] );then
                        cur_order='second'
                        if [[ "$failed_test_lists" == "" ]]; then
                            break
                        else
                            retry_unittests=$(echo "${failed_test_lists}" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
                            retry_unittests_regular=$(echo "$retry_unittests" |awk -F ' ' '{print }' | awk 'BEGIN{ all_str=""}{if (all_str==""){all_str=$1}else{all_str=all_str"$|^"$1}} END{print "^"all_str"$"}')
                        fi
                    elif ( [[ "$exec_times" == "2" ]] );then
                        cur_order='third'
                    fi
                    echo "========================================="
                    echo "This is the ${cur_order} time to re-run"
                    echo "========================================="
                    echo "The following unittest will be re-run:"
                    echo "${retry_unittests}"
                    echo "========================================="
                    rm -f $tmp_dir/*
                    failed_test_lists=''
                    (ctest -R "($retry_unittests_regular)" --output-on-failure -C Release -j 1 | tee $tmpfile ) &
                    wait;
                    collect_failed_tests
                    exec_times=$(echo $exec_times | awk '{print $0+1}')
                done
    else
        # There are more than 30 failed unit tests, so no unit test retry
        is_retry_execuate=1
    fi
    rm -f $tmp_dir/*
}

function show_ut_retry_result() {
    if [[ "$is_retry_execuate" != "0" ]];then
        failed_test_lists_ult=`echo "${failed_test_lists}"`
        echo "========================================="
        echo "There are more than 30 failed unit tests, so no unit test retry!!!"
        echo "========================================="
        echo "${failed_test_lists_ult}"
        exit 8;
    else
        retry_unittests_ut_name=$(echo "$retry_unittests_record" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
        retry_unittests_record_judge=$(echo ${retry_unittests_ut_name}| tr ' ' '\n' | sort | uniq -c | awk '{if ($1 >=3) {print $2}}')
        if [ -z "${retry_unittests_record_judge}" ];then
            echo "========================================"
            echo "There are failed tests, which have been successful after re-run:"
            echo "========================================"
            echo "The following tests have been re-run:"
            echo "${retry_unittests_record}"
        else
            failed_ut_re=$(echo "${retry_unittests_record_judge}" | awk 'BEGIN{ all_str=""}{if (all_str==""){all_str=$1}else{all_str=all_str"|"$1}} END{print all_str}')
            echo "========================================"
            echo "There are failed tests, which have been executed re-run,but success rate is less than 50%:"
            echo "Summary Failed Tests... "
            echo "========================================"
            echo "The following tests FAILED: "
            echo "${retry_unittests_record}" | grep -E "$failed_ut_re"
            exit 8;
        fi
    fi
}

# Step4: Run UT gpu or cpu
set +e
export FLAGS_call_stack_level=2
if [ "${WITH_GPU:-OFF}" == "ON" ];then

    single_ut_mem_0_startTime_s=`date +%s`
    while read line
    do
        run_unittest_gpu "$line" 16
    done < $PADDLE_ROOT/tools/single_card_tests_mem0_new
    single_ut_mem_0_endTime_s=`date +%s`
    single_ut_mem_0_Time_s=`expr $single_ut_mem_0_endTime_s - $single_ut_mem_0_startTime_s`
    echo "ipipe_log_param_1_mem_0_TestCases_Total_Time: $single_ut_mem_0_Time_s s" 

    single_ut_startTime_s=`date +%s`
    while read line
    do
        num=`echo $line | awk -F"$" '{print NF-1}'`
        para_num=`expr $num / 3`
        if [ $para_num -eq 0 ]; then
            para_num=4
        fi
        run_unittest_gpu "$line" $para_num
    done < $PADDLE_ROOT/tools/single_card_tests_new
    single_ut_endTime_s=`date +%s`
    single_ut_Time_s=`expr $single_ut_endTime_s - $single_ut_startTime_s`
    echo "ipipe_log_param_1_TestCases_Total_Time: $single_ut_Time_s s" 

    multiple_ut_mem_0_startTime_s=`date +%s`
    while read line
    do
        run_unittest_gpu "$line" 10
    done < $PADDLE_ROOT/tools/multiple_card_tests_mem0_new
    multiple_ut_mem_0_endTime_s=`date +%s`
    multiple_ut_mem_0_Time_s=`expr $multiple_ut_mem_0_endTime_s - $multiple_ut_mem_0_startTime_s`
    echo "ipipe_log_param_2_mem0_TestCases_Total_Time: $multiple_ut_mem_0_Time_s s" 
    
    multiple_ut_startTime_s=`date +%s`
    while read line
    do
        num=`echo $line | awk -F"$" '{print NF-1}'`
        para_num=`expr $num / 3`
        if [ $para_num -eq 0 ]; then
            para_num=4
        fi
        run_unittest_gpu "$line" $para_num

    done < $PADDLE_ROOT/tools/multiple_card_tests_new
    multiple_ut_endTime_s=`date +%s`
    multiple_ut_Time_s=`expr $multiple_ut_endTime_s - $multiple_ut_startTime_s`
    echo "ipipe_log_param_2_TestCases_Total_Time: $multiple_ut_Time_s s"


    exclusive_ut_mem_0_startTime_s=`date +%s`
    while read line
    do
        run_unittest_gpu "$line" 10
    done < $PADDLE_ROOT/tools/exclusive_card_tests_mem0_new
    exclusive_ut_mem_0_endTime_s=`date +%s`
    exclusive_ut_mem_0_Time_s=`expr $exclusive_ut_mem_0_endTime_s - $exclusive_ut_mem_0_startTime_s`
    echo "ipipe_log_param_-1_mem0_TestCases_Total_Time: $exclusive_ut_mem_0_Time_s s" 

    exclusive_ut_startTime_s=`date +%s`
    while read line
    do
        num=`echo $line | awk -F"$" '{print NF-1}'`
        para_num=`expr $num / 3`
        if [ $para_num -eq 0 ]; then
            para_num=4
        fi
        run_unittest_gpu "$line" $para_num
    done < $PADDLE_ROOT/tools/exclusive_card_tests_new
    exclusive_ut_endTime_s=`date +%s`
    exclusive_ut_Time_s=`expr $exclusive_ut_endTime_s - $exclusive_ut_startTime_s`
    echo "ipipe_log_param_-1_TestCases_Total_Time: $exclusive_ut_Time_s s"

    noparallel_ut_startTime_s=`date +%s`
    while read line
    do
        run_unittest_gpu "$line" 3
    done < $PADDLE_ROOT/tools/no_parallel_case_file
    noparallel_ut_endTime_s=`date +%s`
    noparallel_ut_Time_s=`expr $noparallel_ut_endTime_s - $noparallel_ut_startTime_s`
    echo "ipipe_log_param_noparallel_TestCases_Total_Time: $noparallel_ut_Time_s s"
else
    run_unittest_cpu
fi
collect_failed_tests
set -e
rm -f $tmp_dir/*
if [[ "$failed_test_lists" != "" ]]; then
    unittests_retry
    show_ut_retry_result
fi

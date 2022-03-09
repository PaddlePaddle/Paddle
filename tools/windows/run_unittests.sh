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

set -e
set +x
NIGHTLY_MODE=$1
PRECISION_TEST=$2
WITH_GPU=$3

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

# check added ut

set +e
cp $PADDLE_ROOT/tools/check_added_ut.sh $PADDLE_ROOT/tools/check_added_ut_win.sh
bash $PADDLE_ROOT/tools/check_added_ut_win.sh
rm -rf $PADDLE_ROOT/tools/check_added_ut_win.sh
if [ -f "$PADDLE_ROOT/added_ut" ];then
    added_uts=^$(awk BEGIN{RS=EOF}'{gsub(/\n/,"$|^");print}' $PADDLE_ROOT/added_ut)$
    ctest -R "(${added_uts})" --output-on-failure -C Release --repeat-until-fail 3;added_ut_error=$?
    rm -f $PADDLE_ROOT/added_ut
    if [ "$added_ut_error" != 0 ];then
        echo "========================================"
        echo "Added UT should pass three additional executions"
        echo "========================================"
        exit 8;
    fi
    if nvcc --version | grep 11.2; then
        echo "Only test added_ut temporarily when running in CI-Windows-inference of CUDA 11.2."
        exit 0;
    fi
fi
set -e

# /*==================Fixed Disabled Windows GPU MKL unittests==============================*/
# TODO: fix these unittest that is bound to fail
disable_wingpu_test="^test_model$|\
^test_dataloader_early_reset$|\
^test_add_reader_dependency$|\
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

# /*==================Fixed Disabled Windows GPU MKL unittests==============================*/
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

# /*==================Fixed Disabled Windows GPU inference_api_test unittests==============================*/
disable_win_inference_api_test="^trt_quant_int8_yolov3_r50_test$|\
^test_trt_dynamic_shape_ernie$|\
^test_trt_dynamic_shape_ernie_fp16_ser_deser$|\
^lite_resnet50_test$|\
^test_trt_dynamic_shape_transformer_prune$|\
^lite_mul_model_test$|\
^paddle_infer_api_copy_tensor_tester$"

# /*============================================================================*/

# /*==================Fixed Disabled Windows CPU OPENBLAS unittests==============================*/
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

if [ ${WITH_GPU:-OFF} == "ON" ];then
    export CUDA_VISIBLE_DEVICES=0

    ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d' > all_ut_list
    num=$(ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d' | wc -l)
    echo "Windows 1 card TestCases count is $num"
    if [ ${PRECISION_TEST:-OFF} == "ON" ]; then
        python ${PADDLE_ROOT}/tools/get_pr_ut.py || echo "Failed to obtain ut_list !"
        if [[ -f "ut_list" ]]; then
            echo "PREC length: "`wc -l ut_list`
            precision_cases=`cat ut_list`
            if [[ "$precision_cases" != "" ]];then
                python ${PADDLE_ROOT}/tools/windows/get_prec_ut_list.py
            fi
        fi
    fi

    # sys.argv[1] may exceed max_arg_length when busybox run parallel_UT_rule in windows
    output=$(python ${PADDLE_ROOT}/tools/parallel_UT_rule.py)
    cpu_parallel_job=$(echo $output | cut -d ";" -f 1)
    tetrad_parallel_job=$(echo $output | cut -d ";" -f 2)
    two_parallel_job=$(echo $output | cut -d ";" -f 3)
    non_parallel_job=$(echo $output | cut -d ";" -f 4)
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
        parallel_job=`expr $2 \* $parallel_level_base`
    fi
    echo "************************************************************************"
    echo "********These unittests run $parallel_job job each time with 1 GPU**********"
    echo "************************************************************************"
    export CUDA_VISIBLE_DEVICES=0
    tmpfile=$tmp_dir/$RANDOM
    (ctest -R "$test_case" -E "$disable_ut_quickly|$disable_wingpu_test|$disable_win_trt_test|$long_time_test" -LE "${nightly_label}" --output-on-failure -C Release -j $parallel_job | tee $tmpfile ) &
    wait;
}

function unittests_retry(){
    is_retry_execuate=0
    wintest_error=1
    retry_time=3
    exec_times=0
    exec_retry_threshold=10
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
        # There are more than 10 failed unit tests, so no unit test retry
        is_retry_execuate=1
    fi
    rm -f $tmp_dir/*
}

function show_ut_retry_result() {
    if [[ "$is_retry_execuate" != "0" ]];then
        failed_test_lists_ult=`echo "${failed_test_lists}"`
        echo "========================================="
        echo "There are more than 10 failed unit tests, so no unit test retry!!!"
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

set +e

export FLAGS_call_stack_level=2

if nvcc --version | grep 11.2; then
    echo "Only test added_ut and inference_api_test temporarily when running in CI-Windows-inference of CUDA 11.2."
    export CUDA_VISIBLE_DEVICES=0
    tmpfile=$tmp_dir/$RANDOM
    inference_api_test=^$(ls "paddle/fluid/inference/tests/api" | sed -n 's/\.exe$//pg' | awk BEGIN{RS=EOF}'{gsub(/\n/,"$|^");print}' | sed 's/|\^$//g')
    (ctest -R "$inference_api_test" -E "$disable_win_inference_api_test" --output-on-failure -C Release -j 2 | tee $tmpfile ) &
    wait;
    collect_failed_tests
    set -e
    rm -f $tmp_dir/*
    if [[ "$failed_test_lists" != "" ]]; then
        unittests_retry
        show_ut_retry_result
    fi
    exit 0;
fi

if [ "${WITH_GPU:-OFF}" == "ON" ];then
    run_unittest_gpu $cpu_parallel_job 10
    run_unittest_gpu $tetrad_parallel_job 4
    run_unittest_gpu $two_parallel_job 2
    run_unittest_gpu $non_parallel_job
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

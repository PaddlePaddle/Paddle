# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/utils.sh
init

function parallel_test_base_hybrid() {
    cat <<EOF
    ========================================
    Running unit hybrid tests ...
    ========================================
EOF

#set +x
        ut_startTime_s=`date +%s`
        test_cases=$(ctest -N -V)        # get all test cases
        get_quickly_disable_ut||disable_ut_quickly='disable_ut'   # indicate whether the case was in quickly disable list
        while read -r line; do
            if [[ "$line" == "" ]]; then
                continue
            fi
                matchstr=$(echo $line|grep -oEi 'Test[ \t]+#') || true
                if [[ "$matchstr" == "" ]]; then
                    # Any test case with LABELS property would be parse here
                    # RUN_TYPE=HYBRID mean the case would run in HYBRID CI.
                    is_hybrid=$(echo "$line"|grep -oEi "RUN_TYPE=HYBRID") || true
                    continue
                fi
                testcase=$(echo "$line"|grep -oEi "\w+$")
                if [[ "$is_hybrid" != "" ]]; then
                    if [[ "$eight_cards_tests" == "" ]]; then
                        eight_cards_tests="^$testcase$"
                    else
                        eight_cards_tests="$eight_cards_tests|^$testcase$"
                    fi
                fi
                is_hybrid=''
                matchstr=''
                testcase=''
        done <<< "$test_cases";
        card_test "$eight_cards_tests" -1 1

set -x

        ut_endTime_s=`date +%s`
        echo "HYBRID testCase Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
        if [[ "$EXIT_CODE" != "0" ]]; then
            rm -f $tmp_dir/*
            echo "Summary Failed Tests... "
            echo "========================================"
            echo "The following tests FAILED: "
            echo "${failuretest}" | sort -u
            exit 8;
        fi

    ut_total_endTime_s=`date +%s`
    echo "TestCases Total Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s"
    echo "ipipe_log_param_TestCases_Total_Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
}


function hybrid_paddlex() {
    # PaddleX test
    export DEVICE=($(echo $HIP_VISIBLE_DEVICES | tr "," "\n"))
    export DCU_DEVICES=`echo $HIP_VISIBLE_DEVICES`
    unset HIP_VISIBLE_DEVICES
    git clone --depth=1000 https://gitee.com/paddlepaddle/PaddleX.git
    cd PaddleX
    pip install -e .[base]
    paddlex --install PaddleClas
    paddlex --install PaddleSeg
    wget -q https://paddle-model-ecology.bj.bcebos.com/paddlex/data/cls_flowers_examples.tar -P ./dataset
    tar -xf ./dataset/cls_flowers_examples.tar -C ./dataset/
    wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/seg_optic_examples.tar -P ./dataset
    tar -xf ./dataset/seg_optic_examples.tar -C ./dataset/

    # train Reset50
    echo "Start Reset50"
    python main.py -c paddlex/configs/modules/image_classification/ResNet50.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cls_flowers_examples \
    -o Global.output=resnet50_output \
    -o Global.device="dcu:${DCU_DEVICES}" \
    -o Train.epochs_iters=2

    # inference Reset50
    python main.py -c paddlex/configs/modules/image_classification/ResNet50.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./resnet50_output/best_model/inference" \
    -o Global.device="dcu:${DEVICE[0]}"
    echo "End Reset50"

    echo "Start DeepLabv3+"
    # train DeepLabv3+
    python main.py -c paddlex/configs/modules/semantic_segmentation/Deeplabv3_Plus-R50.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/seg_optic_examples \
    -o Global.output=deeplabv3p_output \
    -o Global.device="dcu:${DCU_DEVICES}" \
    -o Train.epochs_iters=2

    # inference DeepLabv3+
    python main.py -c paddlex/configs/modules/semantic_segmentation/Deeplabv3_Plus-R50.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./deeplabv3p_output/best_model/inference" \
    -o Global.device="dcu:${DEVICE[0]}"
    echo "End DeepLabv3+"
}


function main(){
    cd ${PADDLE_ROOT}/build
    pip install hypothesis
    if ls ${PADDLE_ROOT}/build/python/dist/*whl >/dev/null 2>&1; then
        pip install ${PADDLE_ROOT}/build/python/dist/*whl
    fi
    if ls ${PADDLE_ROOT}/dist/*whl >/dev/null 2>&1; then
        pip install ${PADDLE_ROOT}/dist/*whl
    fi
    cp ${PADDLE_ROOT}/build/test/legacy_test/testsuite.py ${PADDLE_ROOT}/build/python
    cp -r ${PADDLE_ROOT}/build/test/white_list ${PADDLE_ROOT}/build/python
    run_hybrid_ci=${1:-"false"}
    ut_total_startTime_s=`date +%s`

    parallel_test_base_hybrid

    ut_total_endTime_s=`date +%s`
    echo "TestCases Total Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s"
    echo "ipipe_log_param_TestCases_Total_Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt

    if [[ "$IF_DCU" == "ON" ]]; then
      hybrid_paddlex
    fi
}

main

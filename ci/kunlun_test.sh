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

function get_multi_card_ut_list_for_xpu() {
    input_file="${PADDLE_ROOT}/tools/xpu/multi_card_ut_xpu_kl3.local"
    if [ ! -f "$input_file" ]; then
        echo "input file not exist: $input_file"
        exit 102
    fi
    multi_card_ut_list_for_xpu=$(sed 's/^/^/; s/$/$/' "$input_file" | paste -sd'|' -)
    echo "========================================="
    echo "The following unittests are for xpu multi card:"
    echo ${multi_card_ut_list_for_xpu}
    echo "========================================="
}

function parallel_test_base_xpu() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit xpu tests ...
    ========================================
EOF

set +x
        echo "Starting running xpu tests"
        export XPU_OP_LIST_DIR=$tmp_dir
        ut_startTime_s=`date +%s`
        get_quickly_disable_ut||disable_ut_quickly='disable_ut'   # indicate whether the case was in quickly disable list
        get_multi_card_ut_list_for_xpu
        test_cases=$(ctest -N -V -E "$disable_ut_quickly|$multi_card_ut_list_for_xpu")        # cases list which would be run exclusively
        echo "========================================="
        echo "RAW test cases for XPU, already EXCLUDED disable_ut and multi_card:"
        echo ${test_cases}
        echo "========================================="

        single_card_test_num=0
        while read -r line; do
            if [[ "$line" == "" ]]; then
                continue
            fi
            matchstr=$(echo $line|grep -oEi 'Test[ \t]+#') || true
            if [[ "$matchstr" == "" ]]; then
                continue
            fi
            testcase=$(echo "$line"|grep -oEi "\w+$")
            single_card_test_num=$(($single_card_test_num+1))
            if [[ $single_card_test_num -gt 1200 ]]; then
                # too many test cases in single set will lead to ctest "RegularExpression::compile(): Expression too big." error
                # therefore use a new test set
                if [[ "$single_card_tests_1" == "" ]]; then
                    single_card_tests_1="^$testcase$"
                else
                    single_card_tests_1="$single_card_tests_1|^$testcase$"
                fi
                continue
            fi
            if [[ "$single_card_tests" == "" ]]; then
                single_card_tests="^$testcase$"
            else
                single_card_tests="$single_card_tests|^$testcase$"
            fi
        done <<< "$test_cases";

        echo "========================================="
        echo "start to run XPU ut using single card, part 1:"
        echo ${single_card_tests}
        echo "========================================="
        card_test "${single_card_tests}" 1 4

        echo "========================================="
        echo "start to run XPU ut using single card, part 2:"
        echo ${single_card_tests_1}
        echo "========================================="
        card_test "${single_card_tests_1}" 1 4

        echo "========================================="
        echo "start to run XPU ut using multiple cards:"
        echo ${multi_card_ut_list_for_xpu}
        echo "========================================="
        card_test "${multi_card_ut_list_for_xpu}" 2 1

        failed_test_lists=''
        collect_failed_tests
        xputest_error=0
        retry_unittests_record=''
        retry_time=4
        exec_times=0
        exec_time_array=('first' 'second' 'third' 'fourth')
        exec_retry_threshold=10
        is_retry_execute=0
        if [ -n "$failed_test_lists" ];then
            xputest_error=1
            need_retry_ut_str=$(echo "$failed_test_lists" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
            need_retry_ut_arr=(${need_retry_ut_str})
            need_retry_ut_count=${#need_retry_ut_arr[@]}
            retry_unittests=$(echo "$failed_test_lists" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
            if [ $need_retry_ut_count -lt $exec_retry_threshold ];then
                while ( [ $exec_times -lt $retry_time ] )
                    do
                        set +e
                        retry_unittests_record="$retry_unittests_record$failed_test_lists"
                        failed_test_lists_ult=`echo "${failed_test_lists}"`
                        set -e
                        if [[ "${exec_times}" == "1" ]];then
                            if [[ "${failed_test_lists}" == "" ]];then
                                break
                            else
                                retry_unittests=$(echo "$failed_test_lists" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
                            fi
                        fi
                        echo "========================================="
                        echo "This is the ${exec_time_array[$exec_times]} time to re-run"
                        echo "========================================="
                        echo "The following unittest will be re-run:"
                        echo "${retry_unittests}"
                        echo "========================================="

                        retry_unittests_regular=''
                        for line in ${retry_unittests[@]} ;
                            do
                                if [[ "$retry_unittests_regular" == "" ]];then
                                    retry_unittests_regular="^$line$"
                                else
                                    retry_unittests_regular="$retry_unittests_regular|^$line$"
                                fi
                            done
                        rm -f $tmp_dir/*
                        failed_test_lists=''
                        ctest -R "($retry_unittests_regular)" --output-on-failure -j $2 | tee $tmpfile
                        collect_failed_tests
                        exec_times=$[$exec_times+1]
                    done
            else
                # There are more than 10 failed unit tests, so no unit test retry
                is_retry_execute=1
            fi

        fi
        if [[ "$IF_KUNLUN3" == "ON" ]]; then
            export FLAGS_enable_pir_api=0
            #install paddlex
            git clone --depth 1000 https://gitee.com/paddlepaddle/PaddleX.git
            cd PaddleX
            pip install -e .

            #install paddle x dependency
            paddlex --install PaddleClas

            #download paddle dataset
            wget -q https://paddle-model-ecology.bj.bcebos.com/paddlex/data/cls_flowers_examples.tar -P ./dataset
            tar -xf ./dataset/cls_flowers_examples.tar -C ./dataset/

            #train Reset50
            echo "Starting to train ResNet50 model..."
            python main.py -c paddlex/configs/modules/image_classification/ResNet50.yaml \
                -o Global.mode=train \
                -o Global.dataset_dir=./dataset/cls_flowers_examples \
                -o Global.output=resnet50_output \
                -o Global.device="xpu:${CUDA_VISIBLE_DEVICES}"
            echo "Training Resnet50 completed!"

            #inference Reset50
            IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
            echo ${DEVICES[0]}

            echo "Starting to predict ResNet50 model..."
            python main.py -c paddlex/configs/modules/image_classification/ResNet50.yaml \
                -o Global.mode=predict \
                -o Predict.model_dir="./resnet50_output/best_model/inference" \
                -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg" \
                -o Global.device="xpu:${DEVICES[0]}"
            echo "Predicting Resnet50 completed!"
            cd ..
            export FLAGS_enable_pir_api=1
        fi
# set -x
        ut_endTime_s=`date +%s`
        echo "XPU testCase Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
        python ${PADDLE_ROOT}/build/test/xpu/get_test_cover_info.py
        unset XPU_OP_LIST_DIR
        if [ "$xputest_error" != 0 ];then
            show_ut_retry_result
        fi
    fi
}


export PATH=/usr/local/bin:${PATH}
ln -sf $(which python3.10) /usr/local/bin/python
ln -sf $(which pip3.10) /usr/local/bin/pip

mkdir -p ${PADDLE_ROOT}/build
cd ${PADDLE_ROOT}/build

echo "::group::Install python requirements dependencies..."
pip3.10 install --upgrade pip
pip3.10 install -r "${PADDLE_ROOT}/python/requirements.txt"
pip3.10 install -r "${PADDLE_ROOT}/python/unittest_py/requirements.txt"
pip3.10 install PyGithub

pip install hypothesis
echo "::endgroup"

echo "::group::Install paddle wheel"
if ls ${PADDLE_ROOT}/build/python/dist/*whl >/dev/null 2>&1; then
    pip install ${PADDLE_ROOT}/build/python/dist/*whl
fi
if ls ${PADDLE_ROOT}/dist/*whl >/dev/null 2>&1; then
    pip install ${PADDLE_ROOT}/dist/*whl
fi
echo "::endgroup"

cp ${PADDLE_ROOT}/build/test/legacy_test/testsuite.py ${PADDLE_ROOT}/build/python
cp -r ${PADDLE_ROOT}/build/test/white_list ${PADDLE_ROOT}/build/python

ut_total_startTime_s=`date +%s`

parallel_test_base_xpu

ut_total_endTime_s=`date +%s`
echo "TestCases Total Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s"
echo "ipipe_log_param_TestCases_Total_Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt

if [[ -f ${PADDLE_ROOT}/build/build_summary.txt ]];then
echo "=====================build summary======================"
cat ${PADDLE_ROOT}/build/build_summary.txt
echo "========================================================"
fi

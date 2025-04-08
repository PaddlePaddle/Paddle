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

function run_mac_test() {
    export FLAGS_PIR_OPTEST=True
    export FLAGS_CI_PIPELINE=mac
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit tests ...
    ========================================
EOF
        #remove proxy here to fix dist ut 'test_fl_listen_and_serv_op' error on mac.
        #see details: https://github.com/PaddlePaddle/Paddle/issues/24738
        set +x
        my_proxy=$http_proxy
        export http_proxy=
        export https_proxy=
        set -x

        set +ex
        if [ "$1" == "cp38-cp38" ]; then
            pip3.8 uninstall -y paddlepaddle
        elif [ "$1" == "cp39-cp39" ]; then
            pip3.9 uninstall -y paddlepaddle
        elif [ "$1" == "cp310-cp310" ]; then
            pip3.10 uninstall -y paddlepaddle
        elif [ "$1" == "cp311-cp311" ]; then
            pip3.11 uninstall -y paddlepaddle
        elif [ "$1" == "cp312-cp312" ]; then
            pip3.12 uninstall -y paddlepaddle
        fi
        set -ex

        if [ "$1" == "cp38-cp38" ]; then
            pip3.8 install --user ${PADDLE_ROOT}/dist/*.whl
            pip3.8 install --user hypothesis
        elif [ "$1" == "cp39-cp39" ]; then
            pip3.9 install --user ${PADDLE_ROOT}/dist/*.whl
            pip3.9 install --user hypothesis
        elif [ "$1" == "cp310-cp310" ]; then
            pip3.10 install --user ${PADDLE_ROOT}/dist/*.whl
            pip3.10 install --user hypothesis
        elif [ "$1" == "cp311-cp311" ]; then
            pip3.11 install --user ${PADDLE_ROOT}/dist/*.whl
            pip3.11 install --user hypothesis
        elif [ "$1" == "cp312-cp312" ]; then
            pip3.12 install --user ${PADDLE_ROOT}/dist/*.whl
            pip3.12 install --user hypothesis
        fi
        tmpfile_rand=`date +%s%N`
        tmpfile=$tmp_dir/$tmpfile_rand
        set +ex
        ut_startTime_s=`date +%s`
        get_quickly_disable_ut||disable_ut_quickly='disable_ut' # indicate whether the case was in quickly disable list
        if [ ${NIGHTLY_MODE:-OFF} == "ON" ]; then
            nightly_label="(NIGHTLY_LABEL)"
        else
            nightly_label="(RUN_TYPE=NIGHTLY|RUN_TYPE=DIST:NIGHTLY|RUN_TYPE=EXCLUSIVE:NIGHTLY)"
            echo "========================================="
            echo "Unittests with nightly labels  are only run at night"
            echo "========================================="
        fi
        check_approvals_of_unittest 2
        # serial_list: Some single tests need to reduce concurrency
        single_list="^test_cdist$|^test_resnet$|^test_concat_op$|^test_transformer$|^test_bert_with_stride$|^test_paddle_save_load$"
        get_precision_ut_mac
        if [[ "$on_precision" == "0" ]];then
          ctest -E "($disable_ut_quickly|$single_list)" -LE ${nightly_label} --output-on-failure -j $2 | tee $tmpfile
          ctest -R "${single_list}" -E "($disable_ut_quickly)" --output-on-failure -j 1 --timeout 15 | tee -a $tmpfile
        else
            ctest -R "($UT_list_prec)" -E "($disable_ut_quickly)" -LE ${nightly_label} --output-on-failure -j $2 --timeout 15 | tee $tmpfile
            tmpfile_rand=`date +%s%N`
            tmpfile=$tmp_dir/$tmpfile_rand
            ctest -R "($UT_list_prec_1)" -E "(${disable_ut_quickly}|${single_list})" -LE ${nightly_label} --output-on-failure -j $2 --timeout 15 | tee -a $tmpfile
            ctest -R "($single_list)" -E "(${disable_ut_quickly})" --output-on-failure -j 1 --timeout 15 | tee -a $tmpfile
        fi
        failed_test_lists=''
        collect_failed_tests
        mactest_error=0
        retry_unittests_record=''
        retry_time=3
        exec_times=0
        exec_time_array=('first' 'second' 'third')
        exec_retry_threshold=10
        is_retry_execute=0
        if [ -n "$failed_test_lists" ];then
            mactest_error=1
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
                        ctest -R "($retry_unittests_regular)" --output-on-failure -j 4 | tee $tmpfile
                        collect_failed_tests
                        exec_times=$[$exec_times+1]
                    done
            else
                # There are more than 10 failed unit tests, so no unit test retry
                is_retry_execute=1
            fi

        fi
        #mactest_error=$?
        ut_endTime_s=`date +%s`
        echo "Mac testCase Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
        echo "ipipe_log_param_Mac_TestCases_Time: $[ $ut_endTime_s - $ut_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
        paddle version
        # Recovery proxy to avoid failure in later steps
        set +x
        export http_proxy=$my_proxy
        export https_proxy=$my_proxy
        set -x

        if [ "$mactest_error" != 0 ];then
            show_ut_retry_result
        fi
    fi
}

run_mac_test ${PYTHON_ABI:-""} ${PROC_RUN:-1}

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

function run_linux_cpu_test() {
    export FLAGS_PIR_OPTEST=True
    export FLAGS_CI_PIPELINE=py3
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    echo "::group::Installing hypothesis..."
    pip install hypothesis
    echo "::endgroup::"
    if [ -d "${PADDLE_ROOT}/dist/" ]; then
        echo "::group::Installing paddle wheel..."
        pip install ${PADDLE_ROOT}/dist/*whl
        echo "::endgroup::"
    fi
    cp ${PADDLE_ROOT}/build/test/legacy_test/op_test.py ${PADDLE_ROOT}/build/python
    cp ${PADDLE_ROOT}/build/test/legacy_test/testsuite.py ${PADDLE_ROOT}/build/python
    cp -r ${PADDLE_ROOT}/build/test/white_list ${PADDLE_ROOT}/build/python
    ut_total_startTime_s=`date +%s`
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit tests ...
    ========================================
EOF
# set -x
        export TEST_NUM_PERCENT_CASES=0.15
        if [ -a "$PADDLE_ROOT/duplicate_ut" ];then
            duplicate_uts=$(cat $PADDLE_ROOT/duplicate_ut|sed -e 's/\r//g')
            if [[ "$duplicate_uts" != "" ]];then
                set +x
                echo "========================================"
                echo "The new unit test has the same name as the existing unit test"
                cat "$PADDLE_ROOT/duplicate_ut"
                echo "========================================"
                exit 102;
                set -x
            fi
        fi
        if [ -a "$PADDLE_ROOT/added_ut" ];then
            added_uts=^$(awk BEGIN{RS=EOF}'{gsub(/\n/,"$|^");print}' $PADDLE_ROOT/added_ut)$
            ctest -R "(${added_uts})" -LE "RUN_TYPE=DIST|RUN_TYPE=EXCLUSIVE|RUN_TYPE=HYBRID" --output-on-failure --repeat-until-fail 3 --timeout 15;added_ut_error=$?
            ctest -R "(${added_uts})" -L "RUN_TYPE=DIST|RUN_TYPE=EXCLUSIVE" --output-on-failure --repeat-until-fail 3 --timeout 15;added_ut_error_1=$?
            if [ "$added_ut_error" != 0 ] && [ "$added_ut_error_1" != 0 ];then
                echo "========================================"
                echo "Added UT should not exceed 15 seconds"
                echo "========================================"
                exit 8;
            fi
        fi
set +x
        EXIT_CODE=0;

        tmpfile_rand=`date +%s%N`
        tmpfile=$tmp_dir/$tmpfile_rand
        get_quickly_disable_ut||disable_ut_quickly='disable_ut' # indicate whether the case was in quickly disable list
        if [ ${NIGHTLY_MODE:-OFF} == "ON" ]; then
            nightly_label="NIGHTLY_LABEL"
        else
            nightly_label="RUN_TYPE=NIGHTLY|RUN_TYPE=DIST:NIGHTLY|RUN_TYPE=EXCLUSIVE:NIGHTLY"
            echo "========================================="
            echo "Unittests with nightly labels  are only run at night"
            echo "========================================="
        fi
        get_precision_ut_mac
        ut_actual_total_startTime_s=`date +%s`
        if [[ "$on_precision" == "0" ]];then
            ctest -E "$disable_ut_quickly" -LE ${nightly_label} --timeout 120 --output-on-failure -j $2 | tee $tmpfile
        else
            ctest -R "$UT_list_prec" -E "$disable_ut_quickly" -LE ${nightly_label} --timeout 120 --output-on-failure -j $2 | tee $tmpfile
            tmpfile_rand=`date +%s%N`
            tmpfile=$tmp_dir/$tmpfile_rand
            ctest -R "$UT_list_prec_1" -E "$disable_ut_quickly" -LE ${nightly_label} --timeout 120 --output-on-failure -j $2 | tee -a $tmpfile
        fi
        ut_total_endTime_s=`date +%s`
        echo "TestCases Total Time: $[ $ut_total_endTime_s - $ut_actual_total_startTime_s ]s"

        collect_failed_tests
        rm -f $tmp_dir/*
        exec_times=0
        retry_unittests_record=''
        retry_time=2
        exec_time_array=('first' 'second' 'third' 'fourth')
        parallel_failed_tests_exec_retry_threshold=120
        exec_retry_threshold=30
        is_retry_execute=0
        rerun_ut_startTime_s=`date +%s`
        if [ -n "$failed_test_lists" ];then
            EXIT_CODE=1
            if [ ${TIMEOUT_DEBUG_HELP:-OFF} == "ON" ];then
                bash $PADDLE_ROOT/tools/timeout_debug_help.sh "$failed_test_lists"    # cat logs for timeout uts which killed by ctest
            fi
            need_retry_ut_str=$(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/(.\+)//' | sed 's/- //' )
            need_retry_ut_arr=(${need_retry_ut_str})
            need_retry_ut_count=${#need_retry_ut_arr[@]}
            retry_unittests=$(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/(.\+)//' | sed 's/- //' )
            while ( [ $exec_times -lt $retry_time ] )
                do
                    if [[ "${exec_times}" == "0" ]] ;then
                        if [ $need_retry_ut_count -lt $parallel_failed_tests_exec_retry_threshold ];then
                            is_retry_execute=0
                        else
                            is_retry_execute=1
                        fi
                    elif [[ "${exec_times}" == "1" ]] ;then
                        need_retry_ut_str=$(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/(.\+)//' | sed 's/- //' )
                        need_retry_ut_arr=(${need_retry_ut_str})
                        need_retry_ut_count=${#need_retry_ut_arr[@]}
                        if [ $need_retry_ut_count -lt $exec_retry_threshold ];then
                            is_retry_execute=0
                        else
                            is_retry_execute=1
                        fi
                    fi
                    if [[ "$is_retry_execute" == "0" ]];then
                        set +e
                        retry_unittests_record="$retry_unittests_record$failed_test_lists"
                        failed_test_lists_ult=`echo "${failed_test_lists}" |grep -Po '[^ ].*$'`
                        set -e
                        if [[ "${exec_times}" == "1" ]] || [[ "${exec_times}" == "2" ]];then
                            if [[ "${failed_test_lists}" == "" ]];then
                                break
                            else
                                retry_unittests=$(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/(.\+)//' | sed 's/- //' )
                            fi
                        fi
                        echo "========================================="
                        echo "This is the ${exec_time_array[$exec_times]} time to re-run"
                        echo "========================================="
                        echo "The following unittest will be re-run:"
                        echo "${retry_unittests}"
                        retry_unittests_regular=''
                        for line in ${retry_unittests[@]} ;
                            do
                                if [[ "$retry_unittests_regular" == "" ]];then
                                    retry_unittests_regular="^$line$"
                                else
                                    retry_unittests_regular="$retry_unittests_regular|^$line$"
                                fi
                            done
                        failed_test_lists=''
                        ctest -R "$retry_unittests_regular" --timeout 120 --output-on-failure -j 2 | tee $tmpfile
                        collect_failed_tests
                        rm -f $tmp_dir/*
                        exec_times=$[$exec_times+1]
                    else
                        break
                    fi
                done
            retry_unittests_record="$retry_unittests_record$failed_test_lists"
        fi
        rerun_ut_endTime_s=`date +%s`
        echo "ipipe_log_param_Rerun_TestCases_Total_Time: $[ $rerun_ut_endTime_s - $rerun_ut_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
        ut_actual_total_endTime_s=`date +%s`
        echo "ipipe_log_param_actual_TestCases_Total_Time: $[ $ut_actual_total_endTime_s - $ut_actual_total_startTime_s ]s"
        echo "ipipe_log_param_actual_TestCases_Total_Time: $[ $ut_actual_total_endTime_s - $ut_actual_total_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
        if [[ "$EXIT_CODE" != "0" ]]; then
            show_ut_retry_result
        fi
set -ex
    fi
}

run_linux_cpu_test ${PYTHON_ABI:-""} ${PROC_RUN:-1}

if [[ -f ${PADDLE_ROOT}/build/build_summary.txt ]];then
echo "=====================build summary======================"
cat ${PADDLE_ROOT}/build/build_summary.txt
echo "========================================================"
fi

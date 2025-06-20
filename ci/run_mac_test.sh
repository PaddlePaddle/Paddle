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
    tmp_dir='/Users/paddle/tmp'
    mkdir -p $tmp_dir
    mkdir -p ${PADDLE_ROOT}/build
    find $tmp_dir -mindepth 1 -delete
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit tests ...
    ========================================
EOF

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

	if [ "$1" == "cp38-cp38" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.8" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.8/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.8/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.8/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.8/bin/python3
                export PYTHON_INCLUDE_DIR=/Library/Frameworks/Python.framework/Versions/3.8/include/python3.8/
                export PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/3.8/lib/libpython3.8.dylib
                pip3.8 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp39-cp39" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.9" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.9/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.9/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.9/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
                export PYTHON_INCLUDE_DIR=/Library/Frameworks/Python.framework/Versions/3.9/include/python3.9/
                export PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/3.9/lib/libpython3.9.dylib
                pip3.9 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp310-cp310" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.10" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.10/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.10/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.10/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
                export PYTHON_INCLUDE_DIR=/Library/Frameworks/Python.framework/Versions/3.10/include/python3.10/
                export PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/3.10/lib/libpython3.10.dylib
                pip3.10 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp311-cp311" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.11" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.11/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.11/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.11/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
                export PYTHON_INCLUDE_DIR=/Library/Frameworks/Python.framework/Versions/3.11/include/python3.11/
                export PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib
                pip3.11 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp312-cp312" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.12" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.12/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.12/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.12/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
                export PYTHON_INCLUDE_DIR=/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12/
                export PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib
                pip3.12 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
	elif [ "$1" == "cp313-cp313" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.13" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.13/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.13/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.13/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
                export PYTHON_INCLUDE_DIR=/Library/Frameworks/Python.framework/Versions/3.13/include/python3.13/
                export PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/3.13/lib/libpython3.13.dylib
                pip3.13 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
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
        check_approvals_of_unittest
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
			find $tmp_dir -mindepth 1 -delete
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
        ut_endTime_s=`date +%s`
        echo "Mac testCase Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
        echo "ipipe_log_param_Mac_TestCases_Time: $[ $ut_endTime_s - $ut_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt

        if [ "$mactest_error" != 0 ];then
            show_ut_retry_result
        fi
    fi
}

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

# getting quick disable ut list
function get_quickly_disable_ut() {
    python3 -m pip install httpx
    if disable_ut_quickly=$(python3 ${PADDLE_ROOT}/tools/get_quick_disable_lt.py); then
        echo "========================================="
        echo "The following unittests have been disabled:"
        echo ${disable_ut_quickly}
        echo "========================================="
    else

        exit 102
        disable_ut_quickly='disable_ut'
    fi
}

function get_precision_ut_mac() {
    on_precision=0
    UT_list=$(ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d')
    precision_cases=""
    if [ ${PRECISION_TEST:-OFF} == "ON" ]; then
        python3 $PADDLE_ROOT/tools/get_pr_ut.py
        if [[ -f "ut_list" ]]; then
            echo "PREC length: "`wc -l ut_list`
            precision_cases=`cat ut_list`
        fi
    fi
    if [ ${PRECISION_TEST:-OFF} == "ON" ] && [[ "$precision_cases" != "" ]];then
        UT_list_re=''
        on_precision=1
        re=$(cat ut_list|awk -F ' ' '{print }' | awk 'BEGIN{ all_str=""}{if (all_str==""){all_str=$1}else{all_str=all_str"$|^"$1}} END{print "^"all_str"$"}')
        UT_list_prec_1='ut_list_prec2'
        for ut_case in $UT_list; do
            flag=$(echo $ut_case|grep -oE $re)
            if [ -n "$flag" ];then
                if [ -z "$UT_list_prec" ];then
                    UT_list_prec="^$ut_case$"
                elif [[ "${#UT_list_prec}" -gt 10000 ]];then
                    UT_list_prec_1="$UT_list_prec_1|^$ut_case$"
                else
                    UT_list_prec="$UT_list_prec|^$ut_case$"
                fi
            else
                echo ${ut_case} "won't run in PRECISION_TEST mode."
            fi
        done
    fi
}

function show_ut_retry_result() {
    exec_retry_threshold_count=10
    if [[ "$is_retry_execute" != "0" ]]  && [[ "${exec_times}" == "0" ]] ;then
        failed_test_lists_ult=`echo "${failed_test_lists}" | grep -Po '[^ ].*$'`
        echo "========================================="
        echo "There are more than ${exec_retry_threshold_count} failed unit tests in parallel test, so no unit test retry!!!"
        echo "========================================="
        echo "The following tests FAILED: "
        echo "${failed_test_lists_ult}"
        exit 8;
    elif [[ "$is_retry_execute" != "0" ]] && [[ "${exec_times}" == "1" ]];then
        failed_test_lists_ult=`echo "${failed_test_lists}" | grep -Po '[^ ].*$'`
        echo "========================================="
        echo "There are more than 10 failed unit tests, so no unit test retry!!!"
        echo "========================================="
        echo "The following tests FAILED: "
        echo "${failed_test_lists_ult}"
        exit 8;
    else
        retry_unittests_ut_name=$(echo "$retry_unittests_record" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
        retry_unittests_record_judge=$(echo ${retry_unittests_ut_name}| tr ' ' '\n' | sort | uniq -c | awk '{if ($1 >=3) {print $2}}')
        if [ -z "${retry_unittests_record_judge}" ];then
            echo "========================================"
            echo "There are failed tests, which have been successful after re-run:"
            echo "========================================"
            echo "The following tests have been re-ran:"
            echo "${retry_unittests_record}"
        else
            failed_ut_re=$(echo "${retry_unittests_record_judge}" | awk BEGIN{RS=EOF}'{gsub(/\n/,"|");print}')
            echo "========================================"
            echo "There are failed tests, which have been executed re-run,but success rate is less than 50%:"
            echo "Summary Failed Tests... "
            echo "========================================"
            echo "The following tests FAILED: "
            echo "${retry_unittests_record}" | sort -u | grep -E "$failed_ut_re"
            exit 8;
        fi
    fi
}

function check_approvals_of_unittest() {
    set +x
    if [ "$GITHUB_API_TOKEN" == "" ] || [ "$GIT_PR_ID" == "" ]; then
        return 0
    fi

    unittest_spec_diff=`python ${PADDLE_ROOT}/tools/diff_unittest.py ${PADDLE_ROOT}/paddle/fluid/UNITTEST_DEV.spec ${PADDLE_ROOT}/paddle/fluid/UNITTEST_PR.spec`
    if [ "$unittest_spec_diff" != "" ]; then
        approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
        APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 22165420 52485244 32428676 45041955`
        echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
        if [ "${APPROVALS}" == "FALSE" ]; then
            echo "************************************"
            echo -e "It is forbidden to disable or delete the unit-test.\n"
            echo -e "If you must delete it temporarily, please add it to[https://github.com/PaddlePaddle/Paddle/wiki/Temporarily-disabled-Unit-Test]."
            echo -e "Then you must have one RD (kolinwei(recommended), chalsliu, XieYunshen or zhouwei25) approval for the deletion of unit-test. \n"
            echo -e "If you have any problems about deleting unit-test, please read the specification [https://github.com/PaddlePaddle/Paddle/wiki/Deleting-unit-test-is-forbidden]. \n"
            echo -e "Following unit-tests are deleted in this PR: \n ${unittest_spec_diff} \n"
            echo "************************************"
            exit 6
        fi
   fi
}

run_mac_test ${PYTHON_ABI:-""} ${PROC_RUN:-1}

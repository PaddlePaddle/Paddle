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
    disable_ut_quickly='disable_ut'
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

function unittests_retry(){
    parallel_job=4
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
                    (ctest -R "($retry_unittests_regular)" --output-on-failure -C Release -j $parallel_job| tee $tmpfile ) &
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
        failed_test_lists_ult=`echo "${failed_test_lists}" | grep -Po '[^ ].*$'`
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
            echo "The following tests have been re-ran:"
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
(ctest -E "${disable_ut_quickly}" -LE "${nightly_label}" --output-on-failure -C Release -j 8 | tee $tmpfile) &
wait 
collect_failed_tests
set -e
rm -f $tmp_dir/*
if [[ "$failed_test_lists" != "" ]]; then
    unittests_retry
    show_ut_retry_result
fi

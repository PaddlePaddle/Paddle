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

#sot

function init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'

    export PADDLE_ROOT
    if [ -z "${SCRIPT_NAME}" ]; then
        SCRIPT_NAME=$0
    fi

    ENABLE_MAKE_CLEAN=${ENABLE_MAKE_CLEAN:-ON}

    # NOTE(chenweihang): For easy debugging, CI displays the C++ error stacktrace by default
    export FLAGS_call_stack_level=2
}

function build_size() {
    cat <<EOF
    ============================================
    Calculate /paddle/build size and PR whl size
    ============================================
EOF
    if [ "$1" == "paddle_inference" ]; then
        cd ${PADDLE_ROOT}/build
        cp -r paddle_inference_install_dir paddle_inference
        tar -czf paddle_inference.tgz paddle_inference
        buildSize=$(du -h --max-depth=0 ${PADDLE_ROOT}/build/paddle_inference.tgz |awk '{print $1}')
        soLibSize=$(du -h --max-depth=0 ${PADDLE_ROOT}/build/paddle_inference_install_dir/paddle/lib/libpaddle_inference.so |awk '{print $1}')
        echo "Paddle_Inference Size: $buildSize"
        echo "Paddle_Inference Dynamic Library Size: $soLibSize"
        echo "ipipe_log_param_Paddle_Inference_Size: $buildSize" >> ${PADDLE_ROOT}/build/build_summary.txt
        echo "ipipe_log_param_Paddle_Inference_So_Size: $soLibSize" >> ${PADDLE_ROOT}/build/build_summary.txt
    elif [ "$1" == "paddle_inference_c" ]; then
        cd ${PADDLE_ROOT}/build
        cp -r paddle_inference_c_install_dir paddle_inference_c
        tar -czf paddle_inference_c.tgz paddle_inference_c
        buildSize=$(du -h --max-depth=0 ${PADDLE_ROOT}/build/paddle_inference_c.tgz |awk '{print $1}')
        echo "Paddle_Inference Capi Size: $buildSize"
        echo "ipipe_log_param_Paddle_Inference_capi_Size: $buildSize" >> ${PADDLE_ROOT}/build/build_summary.txt
    else
        SYSTEM=`uname -s`
        if [ "$SYSTEM" == "Darwin" ]; then
            com='du -h -d 0'
        else
            com='du -h --max-depth=0'
        fi
        buildSize=$($com ${PADDLE_ROOT}/build |awk '{print $1}')
        echo "Build Size: $buildSize"
        echo "ipipe_log_param_Build_Size: $buildSize" >> ${PADDLE_ROOT}/build/build_summary.txt
        if ls ${PADDLE_ROOT}/build/python/dist/*whl >/dev/null 2>&1; then
            PR_whlSize=$($com ${PADDLE_ROOT}/build/python/dist |awk '{print $1}')
        elif ls ${PADDLE_ROOT}/dist/*whl >/dev/null 2>&1; then
            PR_whlSize=$($com ${PADDLE_ROOT}/dist |awk '{print $1}')
        fi
        echo "PR whl Size: $PR_whlSize"
        echo "ipipe_log_param_PR_whl_Size: $PR_whlSize" >> ${PADDLE_ROOT}/build/build_summary.txt
        PR_soSize=$($com ${PADDLE_ROOT}/build/python/paddle/base/libpaddle.so |awk '{print $1}')
        echo "PR so Size: $PR_soSize"
        echo "ipipe_log_param_PR_so_Size: $PR_soSize" >> ${PADDLE_ROOT}/build/build_summary.txt
    fi
}

function collect_ccache_hits() {
    ccache -s
    ccache_version=$(ccache -V | grep "ccache version" | awk '{print $3}')
    echo "$ccache_version"
    if [[ $ccache_version == 4* ]] ; then
        rate=$(ccache -s | grep "Hits" | awk 'NR==1 {print $5}' | cut -d '(' -f2 | cut -d ')' -f1)
        echo "ccache hit rate: ${rate}%"
    else
        rate=$(ccache -s | grep 'cache hit rate' | awk '{print $4}')
        echo "ccache hit rate: ${rate}"
    fi

    echo "ipipe_log_param_Ccache_Hit_Rate: ${rate}%" >> ${PADDLE_ROOT}/build/build_summary.txt
}

#py3

failed_test_lists=''
tmp_dir=`mktemp -d`

function get_quickly_disable_ut() {
    echo "::group::Installing httpx..."
    python -m pip install httpx
    echo "::endgroup::"
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

function get_precision_ut_mac() {
    on_precision=0
    UT_list=$(ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d')
    precision_cases=""
    if [ ${PRECISION_TEST:-OFF} == "ON" ]; then
        python $PADDLE_ROOT/tools/get_pr_ut.py
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

function show_ut_retry_result() {
    if [ "$SYSTEM" == "Darwin" ]; then
        exec_retry_threshold_count=10
    else
        exec_retry_threshold_count=80
    fi
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
        if [ "$SYSTEM" == "Darwin" ]; then
            retry_unittests_record_judge=$(echo ${retry_unittests_ut_name}| tr ' ' '\n' | sort | uniq -c | awk '{if ($1 >=3) {print $2}}')
        else
            retry_unittests_record_judge=$(echo ${retry_unittests_ut_name}| tr ' ' '\n' | sort | uniq -c | awk '{if ($1 >=2) {print $2}}')
        fi
        if [ -z "${retry_unittests_record_judge}" ];then
            echo "========================================"
            echo "There are failed tests, which have been successful after re-run:"
            echo "========================================"
            echo "The following tests have been re-ran:"
            echo "${retry_unittests_record}"
        else
            failed_ut_re=$(echo "${retry_unittests_record_judge}" | awk BEGIN{RS=EOF}'{gsub(/\n/,"|");print}')
            echo -e "${RED}========================================${NONE}"
            echo -e "${RED}There are failed tests, which have been executed re-run,but success rate is less than 50%:${NONE}"
            echo -e "${RED}Summary Failed Tests... ${NONE}"
            echo -e "${RED}========================================${NONE}"
            echo -e "${RED}The following tests FAILED: ${NONE}"
            echo "${retry_unittests_record}" | sort -u | grep -E "$failed_ut_re" | while IFS= read -r line; do
                echo -e "${RED}${line}${NONE}"
            done
            exit 8;
        fi
    fi
}

function clean_build_files() {
    clean_files=("paddle/fluid/pybind/libpaddle.so" "third_party/flashattn/src/extern_flashattn-build/libflashattn.so" "third_party/install/flashattn/lib/libflashattn.so")

    for file in "${clean_files[@]}"; do
      file=`echo "${PADDLE_ROOT}/build/${file}"`
      if [ -f "$file" ]; then
          rm -rf "$file"
      fi
    done
}


function test_fluid_lib() {
    cat <<EOF
    ========================================
    Testing fluid library for inference ...
    ========================================
EOF
    demo_ci_startTime_s=`date +%s`
    cd ${PADDLE_ROOT}/paddle/fluid/inference/api/demo_ci
    ./run.sh ${PADDLE_ROOT} ${WITH_MKL:-ON} ${WITH_GPU:-OFF} ${INFERENCE_DEMO_INSTALL_DIR} \
             ${WITH_TENSORRT:-ON} ${TENSORRT_ROOT_DIR:-/usr} ${WITH_ONNXRUNTIME:-ON}
    DEMO_EXIT_CODE=$?
    ./clean.sh
    demo_ci_endTime_s=`date +%s`
    echo "demo_ci tests Total time: $[ $demo_ci_endTime_s - $demo_ci_startTime_s ]s"
    echo "ipipe_log_param_Demo_Ci_Tests_Total_Time: $[ $demo_ci_endTime_s - $demo_ci_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt

    infer_ut_startTime_s=`date +%s`
    cd ${PADDLE_ROOT}/test/cpp/inference/infer_ut
    ./run.sh ${PADDLE_ROOT} ${WITH_MKL:-ON} ${WITH_GPU:-OFF} ${INFERENCE_DEMO_INSTALL_DIR} \
             ${TENSORRT_ROOT_DIR:-/usr} ${WITH_ONNXRUNTIME:-ON}
    TEST_EXIT_CODE=$?
    infer_ut_endTime_s=`date +%s`
    echo "infer_ut tests Total time: $[ $infer_ut_endTime_s - $infer_ut_startTime_s ]s"
    echo "ipipe_log_param_Infer_Ut_Tests_Total_Time: $[ $infer_ut_endTime_s - $infer_ut_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
    if [[ "$DEMO_EXIT_CODE" != "0" || "$TEST_EXIT_CODE" != "0" ]]; then
        exit 8;
    fi
}

function test_go_inference_api() {
    cat <<EOF
    ========================================
    Testing go inference api ...
    ========================================
EOF

    # ln paddle_inference_c lib
    cd ${PADDLE_ROOT}/build
    ln -s ${PADDLE_ROOT}/build/paddle_inference_c_install_dir/ ${PADDLE_ROOT}/paddle/fluid/inference/goapi/paddle_inference_c

    # run go test
    cd ${PADDLE_ROOT}/paddle/fluid/inference/goapi
    bash test_action.sh
    EXIT_CODE=$?
    if [[ "$EXIT_CODE" != "0" ]]; then
        exit 8;
    fi
}

function check_approvals_of_unittest() {
    set +x
    if [ "$GITHUB_API_TOKEN" == "" ] || [ "$GIT_PR_ID" == "" ]; then
        return 0
    fi
    # approval_user_list: XiaoguangHu01 46782768,luotao1 6836917,phlrain 43953930,lanxianghit 47554610, zhouwei25 52485244, kolinwei 22165420
    check_times=$1
    if [ $check_times == 1 ]; then
        approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
        if [ "${approval_line}" != "" ]; then
            APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 22165420 52485244`
            echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
            if [ "${APPROVALS}" == "TRUE" ]; then
                echo "==================================="
                echo -e "\n current pr ${GIT_PR_ID} has got approvals. So, Pass CI directly!\n"
                echo "==================================="
                exit 0
            fi
        fi
    elif [ $check_times == 2 ]; then
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
    elif [ $check_times == 3 ]; then
        if [ ${BRANCH} != 'develop' ];then
            return
        fi

        rm -f fluidInference_so_size
        curl -O https://paddle-docker-tar.bj.bcebos.com/paddle_ci_index/fluidInference_so_size
        oriBuildSize=`cat fluidInference_so_size`
        curBuildSize=$(du -m --max-depth=0 ${PADDLE_ROOT}/build/paddle_inference_install_dir/paddle/lib/libpaddle_inference.so |awk '{print $1}')
        diffSize=$(awk "BEGIN{print $curBuildSize-$oriBuildSize}")
        AllDiffSize=$(awk "BEGIN{print $diffSize * 4}")
        cat <<EOF
        ========================================
        Original libpaddle_inference.so Size is ${oriBuildSize}M.
        Current libpaddle_inference.so Size is ${curBuildSize}M.
        In single gpu architecture, Growing size of libpaddle_inference.so is ${diffSize}M.
        In release version, The gpu architecture parameter is "All", The library size is four times to single gpu architecture.
        It means the release version library size growth is about ${AllDiffSize}M.
        ========================================
EOF
        if [ $(awk "BEGIN{print 20<$AllDiffSize}") -eq 1 ] ; then
            approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
            APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 vivienfanghuagood Aurelius84 qingqing01 yuanlehome`
            echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
            if [ "${APPROVALS}" == "FALSE" ]; then
                echo "=========================================================================================="
                echo "This PR make the release inference library size growth exceeds 20 M."
                echo "Then you must have one RD (vivienfanghuagood (Recommend), Aurelius84 (ForPir) qingqing01 or yuanlehome) approval for this PR.\n"
                echo "=========================================================================================="
                exit 6
            fi
        fi
    fi
    set -x
}

function check_excode() {
    if [[ $EXCODE -eq 0 ]];then
        echo "Congratulations!  Your PR passed the paddle-build."
    elif [[ $EXCODE -eq 4 ]];then
        echo "Sorry, your code style check failed."
    elif [[ $EXCODE -eq 5 ]];then
        echo "Sorry, API's example code check failed."
    elif [[ $EXCODE -eq 6 ]];then
        echo "Sorry, your pr need to be approved."
    elif [[ $EXCODE -eq 7 ]];then
        echo "Sorry, build failed."
    elif [[ $EXCODE -eq 8 ]];then
        echo "Sorry, some tests failed."
    elif [[ $EXCODE -eq 9 ]];then
        echo "Sorry, coverage check failed."
    fi
    exit $EXCODE
}

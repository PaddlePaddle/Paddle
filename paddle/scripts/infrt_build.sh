#!/usr/bin/env bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#=================================================
#                   Utils
#=================================================

set -ex

if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

EXIT_CODE=0;
tmp_dir=`mktemp -d`

function init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'

    PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"
    export PADDLE_ROOT
    if [ -z "${SCRIPT_NAME}" ]; then
        SCRIPT_NAME=$0
    fi

    ENABLE_MAKE_CLEAN=${ENABLE_MAKE_CLEAN:-ON}

    # NOTE(chenweihang): For easy debugging, CI displays the C++ error stacktrace by default 
    export FLAGS_call_stack_level=2

    # set CI_SKIP_CPP_TEST if only *.py changed
    # In order to avoid using in some CI(such as daily performance), the current
    # branch must not be `${BRANCH}` which is usually develop.
    if [ ${CI_SKIP_CPP_TEST:-ON} == "OFF"  ];then
        echo "CI_SKIP_CPP_TEST=OFF"
    else
        if [ "$(git branch | grep "^\*" | awk '{print $2}')" != "${BRANCH}" ]; then
            git diff --name-only ${BRANCH} | grep -v "\.py$" || export CI_SKIP_CPP_TEST=ON
        fi
    fi
}

function infrt_gen_and_build() {
    if [ "$1" != "" ]; then
      parallel_number=$1
    fi
    startTime_s=`date +%s`
    set +e
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    rm -f infrt_summary.txt
    cmake ..  -DWITH_MKL=OFF -DWITH_GPU=OFF -DWITH_CRYPTO=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_INFRT=ON -DWITH_PYTHON=OFF -DWITH_TESTING==${WITH_TESTING:-ON}; build_error=$?
    if [ "$build_error" != 0 ];then
        exit 7;
    fi

    make -j ${parallel_number} infrt infrtopt infrt-exec test_infrt_exec trt-exec infrt_lib_dist;build_error=$?
    if [ "$build_error" != 0 ];then
        exit 7;
    fi
    endTime_s=`date +%s`
    [ -n "$startTime_firstBuild" ] && startTime_s=$startTime_firstBuild
    echo "Build Time: $[ $endTime_s - $startTime_s ]s"
    echo "ipipe_log_param_Infrt_Build_Time: $[ $endTime_s - $startTime_s ]s" >> ${PADDLE_ROOT}/build/infrt_summary.txt
}

function test_infrt() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
        cat <<EOF
        ========================================
            Running infrt unit tests ...
        ========================================
EOF
        tmpfile_rand=`date +%s%N`
        tmpfile=$tmp_dir/$tmpfile_rand
        ut_startTime_s=`date +%s`
        ctest --output-on-failure -R test_infrt* | tee $tmpfile
        ut_endTime_s=`date +%s`
        echo "infrt testCase Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
        exit_code=0
        grep -q 'The following tests FAILED:' $tmpfile||exit_code=$?
        if [ $exit_code -eq 0 ]; then
            exit 8;
        fi
    fi
}

function main() {
    local CMD=$1 
    local parallel_number=$2
    init
    case $CMD in
      build_and_test)
        infrt_gen_and_build ${parallel_number}
        test_infrt
        ;;
      build_only)
        infrt_gen_and_build ${parallel_number}
        ;;
      test_only)
        test_infrt
        ;;
      *)
        print_usage
        exit 1
        ;;
      esac
      set +x
      if [[ -f ${PADDLE_ROOT}/build/infrt_summary.txt ]];then
        echo "=====================build summary======================"
        cat ${PADDLE_ROOT}/build/infrt_summary.txt
        echo "========================================================"
      fi
      echo "paddle_build script finished as expected!"
}

main $@

rm -rf $tmp_dir

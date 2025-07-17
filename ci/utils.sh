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

        # github action env variable
        echo "export buildSize=${buildSize}" >> "$HOME/.bashrc"

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
    python -m pip install httpx
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
            retry_unittests_record_judge=$(echo ${retry_unittests_ut_name}| tr ' ' '\n' | sort | uniq -c | awk '{if ($1 >=4) {print $2}}')
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

function determine_kunlun_runner() {
    runner_name=$1

    if [[ $runner_name == "paddle-1" ]]; then
        echo "CUDA_VISIBLE_DEVICES=0,1" >> $GITHUB_ENV
        echo "XPU_CODE_1=/dev/xpu0" >> $GITHUB_ENV
        echo "XPU_CODE_2=/dev/xpu1" >> $GITHUB_ENV
    elif [[ $runner_name == "paddle-2" ]]; then
        echo "CUDA_VISIBLE_DEVICES=0,1" >> $GITHUB_ENV
        echo "XPU_CODE_1=/dev/xpu2" >> $GITHUB_ENV
        echo "XPU_CODE_2=/dev/xpu3" >> $GITHUB_ENV
    elif [[ $runner_name == "paddle-3" ]]; then
        echo "CUDA_VISIBLE_DEVICES=0,1" >> $GITHUB_ENV
        echo "XPU_CODE_1=/dev/xpu4" >> $GITHUB_ENV
        echo "XPU_CODE_2=/dev/xpu5" >> $GITHUB_ENV
    elif [[ $runner_name == "paddle-4" ]]; then
        echo "CUDA_VISIBLE_DEVICES=0,1" >> $GITHUB_ENV
        echo "XPU_CODE_1=/dev/xpu6" >> $GITHUB_ENV
        echo "XPU_CODE_2=/dev/xpu7" >> $GITHUB_ENV
    else
        echo "Unknown runner name: $runner_name"
        exit 1
    fi
    cd $GITHUB_WORKSPACE
}

function determine_dcu_runner() {
    runner_name=$1

    if [[ $runner_name == "paddle-1" ]]; then
        echo "HIP_VISIBLE_DEVICES=0,1" >> $GITHUB_ENV
    elif [[ $runner_name == "paddle-2" ]]; then
        echo "HIP_VISIBLE_DEVICES=2,3" >> $GITHUB_ENV
    elif [[ $runner_name == "paddle-3" ]]; then
        echo "HIP_VISIBLE_DEVICES=4,5" >> $GITHUB_ENV
    elif [[ $runner_name == "paddle-4" ]]; then
        echo "HIP_VISIBLE_DEVICES=6,7" >> $GITHUB_ENV
    else
        echo "Unknown runner name: $runner_name"
        exit 1
    fi
    cd $GITHUB_WORKSPACE
    # rm -rf * .[^.]* .??*
}

function determine_npu_runner() {
    runner_name=$1
    if [[ $runner_name == "paddle-1" ]]; then
        echo "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3" >> $GITHUB_ENV
    elif [[ $runner_name == "paddle-2" ]]; then
        echo "ASCEND_RT_VISIBLE_DEVICES=4,5,6,7" >> $GITHUB_ENV
    elif [[ $runner_name == "paddle-3" ]]; then
        echo "ASCEND_RT_VISIBLE_DEVICES=8,9,10,11" >> $GITHUB_ENV
    elif [[ $runner_name == "paddle-4" ]]; then
        echo "ASCEND_RT_VISIBLE_DEVICES=12,13,14,15" >> $GITHUB_ENV
    else
        echo "Unknown runner name: $runner_name"
        exit 1
    fi
}

function cmake_base() {
    # Build script will not fail if *.deb does not exist
    rm *.deb 2>/dev/null || true
    # Delete previous built whl packages
    rm -rf python/dist 2>/dev/null || true

    # Delete previous built paddle cache
    rm -rf python/paddle 2>/dev/null || true

    # Support build for all python3 versions
    PYTHON_FLAGS=""
    SYSTEM=`uname -s`
    if [ "$SYSTEM" == "Darwin" ]; then
        echo "Using python abi: $1"
        if [ "$1" == "cp38-cp38" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.8" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.8/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.8/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.8/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.8/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.8/include/python3.8/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.8/lib/libpython3.8.dylib"
                pip3.8 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp39-cp39" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.9" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.9/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.9/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.9/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.9/include/python3.9/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.9/lib/libpython3.9.dylib"
                pip3.9 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp310-cp310" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.10" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.10/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.10/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.10/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.10/include/python3.10/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.10/lib/libpython3.10.dylib"
                pip3.10 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp311-cp311" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.11" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.11/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.11/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.11/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.11/include/python3.11/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib"
                pip3.11 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp312-cp312" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.12" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.12/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.12/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.12/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib"
                pip3.12 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
	elif [ "$1" == "cp313-cp313" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.13" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.13/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.13/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.13/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.13/include/python3.13/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.13/lib/libpython3.13.dylib"
                pip3.13 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        fi
    else
        if [ "$1" != "" ]; then
            echo "using python abi: $1"
            if [ "$1" == "cp38-cp38" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.8.0/bin/python3.8
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.8.0/include/python3.8
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.8.0/lib/libpython3.so"
                pip3.8 install -r ${PADDLE_ROOT}/python/requirements.txt
                pip3.8 install -r ${PADDLE_ROOT}/paddle/scripts/compile_requirements.txt
            elif [ "$1" == "cp39-cp39" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.9.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.9.0/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.9.0/bin/python3.9
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.9.0/include/python3.9
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.9.0/lib/libpython3.so"
                pip3.9 install -r ${PADDLE_ROOT}/python/requirements.txt
                pip3.9 install -r ${PADDLE_ROOT}/paddle/scripts/compile_requirements.txt
            elif [ "$1" == "cp310-cp310" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.10.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.10.0/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.10.0/bin/python3.10
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.10.0/include/python3.10
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.10.0/lib/libpython3.so"
                pip3.10 install -r ${PADDLE_ROOT}/python/requirements.txt
                pip3.10 install -r ${PADDLE_ROOT}/paddle/scripts/compile_requirements.txt
            elif [ "$1" == "cp311-cp311" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.11.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.11.0/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.11.0/bin/python3.11
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.11.0/include/python3.11
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.11.0/lib/libpython3.so"
                pip3.11 install -r ${PADDLE_ROOT}/python/requirements.txt
                pip3.11 install -r ${PADDLE_ROOT}/paddle/scripts/compile_requirements.txt
            elif [ "$1" == "cp312-cp312" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.12.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.12.0/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.12.0/bin/python3.12
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.12.0/include/python3.12
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.12.0/lib/libpython3.so"
                pip3.12 install -r ${PADDLE_ROOT}/python/requirements.txt
                pip3.12 install -r ${PADDLE_ROOT}/paddle/scripts/compile_requirements.txt
	    elif [ "$1" == "cp313-cp313" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.13.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.13.0/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.13.0/bin/python3.13
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.13.0/include/python3.13
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.13.0/lib/libpython3.so"
                pip3.13 install -r ${PADDLE_ROOT}/python/requirements.txt
                pip3.13 install -r ${PADDLE_ROOT}/paddle/scripts/compile_requirements.txt
            elif [ "$1" == "conda-python3.8" ]; then
                export LD_LIBRARY_PATH=/opt/conda/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/conda/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/conda/bin/python
                                     -DPYTHON_INCLUDE_DIR:PATH=/opt/conda/include/python3.8m
                                     -DPYTHON_LIBRARIES:FILEPATH=/opt/conda/lib/libpython3.so"
                /opt/conda/bin/pip install -r ${PADDLE_ROOT}/python/requirements.txt
            fi
            # for CINN, to find libcuda.so.1
            export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11.2/compat/
        else
            pip install -r ${PADDLE_ROOT}/python/requirements.txt
        fi
    fi

    if [ "$SYSTEM" == "Darwin" ]; then
        WITH_DISTRIBUTE="OFF"
        WITH_AVX=${WITH_AVX:-ON}
        WITH_ARM=${WITH_ARM:-OFF}
        INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR:-~/.cache/inference_demo}
    else
        INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR:-/root/.cache/inference_demo}
    fi

    distributed_flag=${WITH_DISTRIBUTE:-OFF}
    gloo_flag=${distributed_flag}
    pscore_flag=${distributed_flag}
    pslib_flag=${WITH_PSLIB:-OFF}
    if [ "${pslib_flag}" == "ON" ];then
      pscore_flag=${WITH_PSCORE:-OFF}
    fi

    if [ "$2" != "approval" ];then
      which python
      python -V
      python -m pip install distro
      python ${PADDLE_ROOT}/tools/summary_env.py
      bash ${PADDLE_ROOT}/tools/get_cpu_info.sh
    fi

    cat <<EOF
    ========================================
    Configuring cmake in /paddle/build ...
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
        ${PYTHON_FLAGS}
        -DWITH_GPU=${WITH_GPU:-OFF}
        -DWITH_TENSORRT=${WITH_TENSORRT:-ON}
        -DWITH_OPENVINO=${WITH_OPENVINO:-OFF}
        -DWITH_ROCM=${WITH_ROCM:-OFF}
        -DWITH_CINN=${WITH_CINN:-OFF}
        -DWITH_DISTRIBUTE=${distributed_flag}
        -DWITH_MKL=${WITH_MKL:-ON}
        -DWITH_AVX=${WITH_AVX:-OFF}
        -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All}
        -DNEW_RELEASE_PYPI=${NEW_RELEASE_PYPI:-OFF}
        -DNEW_RELEASE_ALL=${NEW_RELEASE_ALL:-OFF}
        -DNEW_RELEASE_JIT=${NEW_RELEASE_JIT:-OFF}
        -DWITH_PYTHON=${WITH_PYTHON:-ON}
        -DCUDNN_ROOT=/usr/
        -DWITH_TESTING=${WITH_TESTING:-ON}
        -DWITH_CPP_TEST=${WITH_CPP_TEST:-ON}
        -DWITH_COVERAGE=${WITH_COVERAGE:-OFF}
        -DWITH_INCREMENTAL_COVERAGE=${WITH_INCREMENTAL_COVERAGE:-OFF}
        -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DWITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON}
        -DINFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR}
        -DPY_VERSION=${PY_VERSION:-3.8}
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-/paddle/build}
        -DWITH_PSCORE=${pscore_flag}
        -DWITH_PSLIB=${pslib_flag}
        -DWITH_GLOO=${gloo_flag}
        -DWITH_XPU=${WITH_XPU:-OFF}
        -DWITH_IPU=${WITH_IPU:-OFF}
        -DWITH_UNITY_BUILD=${WITH_UNITY_BUILD:-OFF}
        -DWITH_XPU_BKCL=${WITH_XPU_BKCL:-OFF}
        -DWITH_XPU_XHPC=${WITH_XPU_XHPC:-OFF}
        -DWITH_XPU_XFT=${WITH_XPU_XFT:-OFF}
        -DWITH_XPU_XRE5=${WITH_XPU_XRE5:-OFF}
        -DWITH_XPU_FFT=${WITH_XPU_FFT:-OFF}
        -DWITH_ARM=${WITH_ARM:-OFF}
        -DWITH_STRIP=${WITH_STRIP:-ON}
        -DON_INFER=${ON_INFER:-OFF}
        -DWITH_HETERPS=${WITH_HETERPS:-OFF}
        -DWITH_RECORD_BUILDTIME=${WITH_RECORD_BUILDTIME:-OFF}
        -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}"
        -DWITH_ONNXRUNTIME=${WITH_ONNXRUNTIME:-OFF}
        -DWITH_NVCC_LAZY=${WITH_NVCC_LAZY:-ON}
        -DWITH_CUDNN_FRONTEND=${WITH_CUDNN_FRONTEND:-OFF}
        -DFA_BUILD_WITH_CACHE=${FA_BUILD_WITH_CACHE:ON}
    ========================================
EOF
    # Disable UNITTEST_USE_VIRTUALENV in docker because
    # docker environment is fully controlled by this script.
    # See /Paddle/CMakeLists.txt, UNITTEST_USE_VIRTUALENV option.
    set +e
    cmake .. \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} \
        ${PYTHON_FLAGS} \
        -DWITH_GPU=${WITH_GPU:-OFF} \
        -DWITH_TENSORRT=${WITH_TENSORRT:-ON} \
        -DWITH_OPENVINO=${WITH_OPENVINO:-OFF} \
        -DWITH_ROCM=${WITH_ROCM:-OFF} \
        -DWITH_CINN=${WITH_CINN:-OFF} \
        -DWITH_DISTRIBUTE=${distributed_flag} \
        -DWITH_MKL=${WITH_MKL:-ON} \
        -DWITH_AVX=${WITH_AVX:-OFF} \
        -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All} \
        -DNEW_RELEASE_PYPI=${NEW_RELEASE_PYPI:-OFF} \
        -DNEW_RELEASE_ALL=${NEW_RELEASE_ALL:-OFF} \
        -DNEW_RELEASE_JIT=${NEW_RELEASE_JIT:-OFF} \
        -DWITH_PYTHON=${WITH_PYTHON:-ON} \
        -DCUDNN_ROOT=/usr/ \
        -DWITH_TESTING=${WITH_TESTING:-ON} \
        -DWITH_CPP_TEST=${WITH_CPP_TEST:-ON} \
        -DWITH_COVERAGE=${WITH_COVERAGE:-OFF} \
        -DWITH_INCREMENTAL_COVERAGE=${WITH_INCREMENTAL_COVERAGE:-OFF} \
        -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DWITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON} \
        -DINFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR} \
        -DPY_VERSION=${PY_VERSION:-3.8} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-/paddle/build} \
        -DWITH_PSCORE=${pscore_flag} \
        -DWITH_PSLIB=${pslib_flag} \
        -DWITH_GLOO=${gloo_flag} \
        -DWITH_XPU=${WITH_XPU:-OFF} \
	    -DWITH_XPU_XRE5=${WITH_XPU_XRE5:-OFF} \
        -DWITH_IPU=${WITH_IPU:-OFF} \
        -DXPU_SDK_ROOT=${XPU_SDK_ROOT:-""} \
        -DWITH_XPU_BKCL=${WITH_XPU_BKCL:-OFF} \
        -DWITH_XPU_XHPC=${WITH_XPU_XHPC:-OFF} \
        -DWITH_XPU_XFT=${WITH_XPU_XFT:-OFF} \
        -DWITH_XPU_FFT=${WITH_XPU_FFT:-OFF} \
        -DWITH_ARM=${WITH_ARM:-OFF} \
        -DWITH_STRIP=${WITH_STRIP:-ON} \
        -DON_INFER=${ON_INFER:-OFF} \
        -DWITH_HETERPS=${WITH_HETERPS:-OFF} \
        -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
        -DWITH_RECORD_BUILDTIME=${WITH_RECORD_BUILDTIME:-OFF} \
        -DWITH_UNITY_BUILD=${WITH_UNITY_BUILD:-OFF}  \
        -DWITH_ONNXRUNTIME=${WITH_ONNXRUNTIME:-OFF}  \
        -DWITH_NVCC_LAZY=${WITH_NVCC_LAZY:-ON} \
        -DFA_BUILD_WITH_CACHE=${FA_BUILD_WITH_CACHE:ON} \
        -DWITH_CUDNN_FRONTEND=${WITH_CUDNN_FRONTEND:-OFF};build_error=$?

    if [ "$build_error" != 0 ];then
        return 7;
    fi
}

function check_approvals_of_unittest() {
    set +x
    if [ "$GITHUB_TOKEN" == "" ] || [ "$GIT_PR_ID" == "" ]; then
        return 0
    fi
    # approval_user_list: XiaoguangHu01 46782768,luotao1 6836917,phlrain 43953930,lanxianghit 47554610, zhouwei25 52485244, kolinwei 22165420
    check_times=$1
    if [ $check_times == 1 ]; then
        approval_line=`curl -H "Authorization: token ${GITHUB_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
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
            approval_line=`curl -H "Authorization: token ${GITHUB_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
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

function check_cinn_file_diff() {
    CINN_FILE_LIST=(
        CMakeLists.txt
        cmake
        paddle/cinn
        python/cinn
        python/CMakeLists.txt
        python/setup_cinn.py.in
        test/CMakeLists.txt
        test/cinn
        test/cpp/cinn
        tools/cinn
    )

    run_cinn_ut="OFF"
    for change_fie in $(git diff --name-only upstream/${BRANCH});
    do
      for cinn_file in ${CINN_FILE_LIST[@]};
      do
        if [[ ${change_fie} =~ ^"${cinn_file}".* ]]; then
          run_cinn_ut="ON"
          break
        fi
      done
      if [[ "ON" == ${run_cinn_ut} ]]; then
        break
      fi
    done
    echo $run_cinn_ut
}

function case_count(){
    cat <<EOF
    ============================================
    Generating TestCases Count ...
    ============================================
EOF
    testcases=$1
    num=$(echo $testcases|grep -o '\^'|wc -l)
    if [[ "$2" == "-1" ]]; then
        echo "exclusive TestCases count is $num"
        echo "ipipe_log_param_Exclusive_TestCases_Count: $num" >> ${PADDLE_ROOT}/build/build_summary.txt
    else
        echo "$2 card TestCases count is $num"
        echo "ipipe_log_param_${2}_Cards_TestCases_Count: $num" >> ${PADDLE_ROOT}/build/build_summary.txt
    fi
}

EXIT_CODE=0
function caught_error() {
 for job in `jobs -p`; do
        # echo "PID => ${job}"
        if ! wait ${job} ; then
            JOB_EXIT_CODE=$?
            echo "At least one test failed with exit code => ${JOB_EXIT_CODE}" ;
            EXIT_CODE=1;
        fi
    done
}

function card_test() {
    set -m
    case_count $1 $2
    ut_startTime_s=`date +%s`

    testcases=$1
    cardnumber=$2
    parallel_level_base=${CTEST_PARALLEL_LEVEL:-1}

    # run ut based on the label
    if [[ "${UT_RUN_TYPE_SETTING}" == "INFER" ]];then
        run_label_mode="-L (RUN_TYPE=INFER)"
    elif [[ "${UT_RUN_TYPE_SETTING}" == "DIST" ]];then
        run_label_mode="-L (RUN_TYPE=DIST|RUN_TYPE=EXCLUSIVE)"
    elif [[ "${UT_RUN_TYPE_SETTING}" == "WITHOUT_INFER" ]];then
        run_label_mode="-LE (RUN_TYPE=INFER)"
    elif [[ "${UT_RUN_TYPE_SETTING}" == "WITHOUT_HYBRID" ]];then
        run_label_mode="-LE (RUN_TYPE=HYBRID)"
    elif [[ "${UT_RUN_TYPE_SETTING}" == "OTHER" ]];then
        run_label_mode="-LE (RUN_TYPE=INFER|RUN_TYPE=DIST|RUN_TYPE=EXCLUSIVE)"
    fi

    # get the CUDA device count, XPU device count is one
    if [ "${WITH_XPU}" == "ON" ];then
        CUDA_DEVICE_COUNT=1
    elif [ "${WITH_ROCM}" == "ON" ];then
        CUDA_DEVICE_COUNT=$(rocm-smi -i | grep DCU | wc -l)
    elif [ "${WITH_IPU}" == "ON" ];then
        CUDA_DEVICE_COUNT=1
    else
        CUDA_DEVICE_COUNT=$(nvidia-smi -L | wc -l)
    fi

    if (( $cardnumber == -1 ));then
        cardnumber=$CUDA_DEVICE_COUNT
    fi

    if (( $# > 2 )); then
        parallel_job=`expr $3 \* $parallel_level_base`
    else
        parallel_job=$parallel_level_base
    fi

    if [[ "$testcases" == "" ]]; then
        return 0
    fi

    trap 'caught_error' CHLD
    tmpfile_rand=`date +%s%N`
    NUM_PROC=$[CUDA_DEVICE_COUNT/$cardnumber]
    echo "****************************************************************"
    echo "***These unittests run $parallel_job job each time with $cardnumber GPU***"
    echo "****************************************************************"
    for (( i = 0; i < $NUM_PROC; i++ )); do
        # CUDA_VISIBLE_DEVICES http://acceleware.com/blog/cudavisibledevices-masking-gpus
        # ctest -I https://cmake.org/cmake/help/v3.0/manual/ctest.1.html?highlight=ctest
        cuda_list=()
        for (( j = 0; j < cardnumber; j++ )); do
            if [ $j -eq 0 ]; then
                    cuda_list=("$[i*cardnumber]")
                else
                    cuda_list="$cuda_list,$[i*cardnumber+j]"
            fi
        done
        tmpfile=$tmp_dir/$tmpfile_rand"_"$i
        if [ ${TESTING_DEBUG_MODE:-OFF} == "ON" ] ; then
            if [[ $cardnumber == $CUDA_DEVICE_COUNT ]]; then
                (ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" ${run_label_mode} -V --timeout 120 -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
            else
                if [ "$WITH_ROCM" == "ON" ];then
                    (env HIP_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" ${run_label_mode} --timeout 120 -V -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
                else
                    (env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" ${run_label_mode} --timeout 120 -V -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
                fi
            fi
        else
            if [[ $cardnumber == $CUDA_DEVICE_COUNT ]]; then
                (ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" ${run_label_mode} --timeout 120 --output-on-failure  -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
            else
                if [ "$WITH_ROCM" == "ON" ];then
                    (env HIP_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" ${run_label_mode} --timeout 120 --output-on-failure  -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
                else
                    (env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" ${run_label_mode} --timeout 120 --output-on-failure  -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
                fi

            fi
        fi
    done
    wait; # wait for all subshells to finish
    trap - CHLD
    ut_endTime_s=`date +%s`
    if (( $2 == -1 )); then
        echo "exclusive TestCases Total Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
    else
        echo "$2 card TestCases Total Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
    fi
    echo "$2 card TestCases finished!!!! "
    set +m
}

function parallel_test_base_gpu_test() {
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit tests in parallel way ...
    ========================================
EOF


set -x
        # set trt_convert ut to run 15% cases.
        export TEST_NUM_PERCENT_CASES=0.15
        export FLAGS_trt_ibuilder_cache=1
        precision_cases=""
        #check change of pr_unittests and dev_unittests
        check_approvals_of_unittest 2
        ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d' > ${PADDLE_ROOT}/build/all_ut_list
        if [ ${PRECISION_TEST:-OFF} == "ON" ]; then
            python $PADDLE_ROOT/tools/get_pr_ut.py
        fi
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
            if [ "$WITH_ROCM" == "ON" ];then
                env HIP_VISIBLE_DEVICES=0 ctest -R "(${added_uts})" -LE "RUN_TYPE=DIST|RUN_TYPE=EXCLUSIVE|RUN_TYPE=HYBRID" --output-on-failure --repeat-until-fail 3 --timeout 20;added_ut_error=$?
            else
                env CUDA_VISIBLE_DEVICES=0 ctest -R "(${added_uts})" -LE "RUN_TYPE=DIST|RUN_TYPE=EXCLUSIVE|RUN_TYPE=HYBRID" --output-on-failure --repeat-until-fail 3 --timeout 20;added_ut_error=$?
            fi
            ctest -R "(${added_uts})" -L "RUN_TYPE=DIST|RUN_TYPE=EXCLUSIVE" --output-on-failure --repeat-until-fail 3 --timeout 20;added_ut_error_1=$?
            if [ "$added_ut_error" != 0 ] && [ "$added_ut_error_1" != 0 ];then
                echo "========================================"
                echo "Added UT should not exceed 20 seconds"
                echo "========================================"
                exit 8;
            fi
        fi
set +x
        EXIT_CODE=0;
        wget --no-proxy https://paddle-docker-tar.bj.bcebos.com/pre_test/CTestCostData.txt --no-check-certificate
        mkdir -p ${PADDLE_ROOT}/build/Testing/Temporary/
        cp -r ${PADDLE_ROOT}/build/CTestCostData.txt ${PADDLE_ROOT}/build/Testing/Temporary/

        get_quickly_disable_ut||disable_ut_quickly='disable_ut'    # indicate whether the case was in quickly disable list
        test_cases=$(ctest -N -V) # get all test cases

        if [ ${WITH_CINN:-OFF} == "ON" ]; then
            pushd ${PADDLE_ROOT}/build/paddle/cinn
            ctest -N -E "test_frontend_interpreter" | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d' > ${PADDLE_ROOT}/build/pr_ci_cinn_gpu_ut_list
            popd
            pushd ${PADDLE_ROOT}/build/test/cpp/cinn # Note: Some tests have been moved to test/cpp/cinn.
            ctest -N -E "test_frontend_interpreter" | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d' >> ${PADDLE_ROOT}/build/pr_ci_cinn_gpu_ut_list
            popd
            ctest -N -L "RUN_TYPE=CINN" | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d' > ${PADDLE_ROOT}/build/pr_ci_cinn_ut_list
            echo "========================================"
            echo "::group::pr_ci_cinn_ut_list: "
            cat ${PADDLE_ROOT}/build/pr_ci_cinn_ut_list
            echo "::endgroup::"
            echo "========================================"
            echo "::group::pr_ci_cinn_gpu_ut_list: "
            cat ${PADDLE_ROOT}/build/pr_ci_cinn_gpu_ut_list
            echo "::endgroup::"
            echo "========================================"
        fi

        python ${PADDLE_ROOT}/tools/group_case_for_parallel.py ${PADDLE_ROOT}

        if [ ${WITH_CINN=-OFF} == "ON" ]; then
            run_cinn_ut=`check_cinn_file_diff`
            if [[ "OFF" == ${run_cinn_ut} ]]; then
              echo -e "${BLUE}No CINN-related changes were found${NONE}"
              echo -e "${BLUE}Skip PR-CI-CINN-GPU UT CI${NONE}"
            else
                # run pr-ci-cinn-gpu ut
                cinn_gpu_ut_startTime_s=`date +%s`
                while read line
                do
                    card_test "$line" 1
                done < $PADDLE_ROOT/tools/new_pr_ci_cinn_gpu_ut_list
                cinn_gpu_ut_endTime_s=`date +%s`
                echo "ipipe_log_param_cinn_gpu_TestCases_Total_Time: $[ $cinn_gpu_ut_endTime_s - $cinn_gpu_ut_startTime_s ]s"
                echo "ipipe_log_param_cinn_gpu_TestCases_Total_Time: $[ $cinn_gpu_ut_endTime_s - $cinn_gpu_ut_startTime_s ]s"  >> ${PADDLE_ROOT}/build/build_summary.txt
            fi

            # run pr-ci-cinn ut
            cinn_ut_startTime_s=`date +%s`
            while read line
            do
                card_test "$line" 1
            done < $PADDLE_ROOT/tools/new_pr_ci_cinn_ut_list
            cinn_ut_endTime_s=`date +%s`
            echo "ipipe_log_param_cinn_TestCases_Total_Time: $[ $cinn_ut_endTime_s - $cinn_ut_startTime_s ]s"
            echo "ipipe_log_param_cinn_TestCases_Total_Time: $[ $cinn_ut_endTime_s - $cinn_ut_startTime_s ]s"  >> ${PADDLE_ROOT}/build/build_summary.txt
        fi

        single_ut_mem_0_startTime_s=`date +%s`
        while read line
        do
            card_test "$line" 1 4
        done < $PADDLE_ROOT/tools/single_card_tests_mem0_new
        single_ut_mem_0_endTime_s=`date +%s`
        echo "ipipe_log_param_1_mem_0_TestCases_Total_Time: $[ $single_ut_mem_0_endTime_s - $single_ut_mem_0_startTime_s ]s"
        echo "ipipe_log_param_1_mem_0_TestCases_Total_Time: $[ $single_ut_mem_0_endTime_s - $single_ut_mem_0_startTime_s ]s"  >> ${PADDLE_ROOT}/build/build_summary.txt

        single_ut_startTime_s=`date +%s`
        while read line
        do
            num=$[(`echo $line | awk -F"$" '{print NF-1}'`-1)/6]
            if [ $num -eq 0 ]; then
                num=1
            fi
            card_test "$line" 1 $num
        done < $PADDLE_ROOT/tools/single_card_tests_new
        single_ut_endTime_s=`date +%s`
        echo "ipipe_log_param_1_TestCases_Total_Time: $[ $single_ut_endTime_s - $single_ut_startTime_s ]s"
        echo "ipipe_log_param_1_TestCases_Total_Time: $[ $single_ut_endTime_s - $single_ut_startTime_s ]s"   >> ${PADDLE_ROOT}/build/build_summary.txt

        multiple_ut_mem_0_startTime_s=`date +%s`
        while read line
        do
            card_test "$line" 2 4
        done < $PADDLE_ROOT/tools/multiple_card_tests_mem0_new
        multiple_ut_mem_0_endTime_s=`date +%s`
        echo "ipipe_log_param_2_mem0_TestCases_Total_Time: $[ $multiple_ut_mem_0_endTime_s - $multiple_ut_mem_0_startTime_s ]s"
        echo "ipipe_log_param_2_mem0_TestCases_Total_Time: $[ $multiple_ut_mem_0_endTime_s - $multiple_ut_mem_0_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
        multiple_ut_startTime_s=`date +%s`
        while read line
        do
            num=$[(`echo $line | awk -F"$" '{print NF-1}'`-1)/6]
            if [ $num -eq 0 ]; then
                num=1
            fi
            card_test "$line" 2 $num

        done < $PADDLE_ROOT/tools/multiple_card_tests_new
        multiple_ut_endTime_s=`date +%s`
        echo "ipipe_log_param_2_TestCases_Total_Time: $[ $multiple_ut_endTime_s - $multiple_ut_startTime_s ]s"
        echo "ipipe_log_param_2_TestCases_Total_Time: $[ $multiple_ut_endTime_s - $multiple_ut_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt

        exclusive_ut_mem_0_startTime_s=`date +%s`
        while read line
        do
            card_test "$line" -1 4
        done < $PADDLE_ROOT/tools/exclusive_card_tests_mem0_new
        exclusive_ut_mem_0_endTime_s=`date +%s`
        echo "ipipe_log_param_-1_mem0_TestCases_Total_Time: $[ $exclusive_ut_mem_0_endTime_s - $exclusive_ut_mem_0_startTime_s ]s"
        echo "ipipe_log_param_-1_mem0_TestCases_Total_Time: $[ $exclusive_ut_mem_0_endTime_s - $exclusive_ut_mem_0_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt

        exclusive_ut_startTime_s=`date +%s`
        while read line
        do
            num=$[(`echo $line | awk -F"$" '{print NF-1}'`-1)/6]
            if [ $num -eq 0 ]; then
                num=1
            fi
            card_test "$line" -1 $num
        done < $PADDLE_ROOT/tools/exclusive_card_tests_new
        exclusive_ut_endTime_s=`date +%s`
        echo "ipipe_log_param_-1_TestCases_Total_Time: $[ $exclusive_ut_endTime_s - $exclusive_ut_startTime_s ]s"
        echo "ipipe_log_param_-1_TestCases_Total_Time: $[ $exclusive_ut_endTime_s - $exclusive_ut_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt

        noparallel_ut_startTime_s=`date +%s`
        while read line
        do
            card_test "$line" -1 4
        done < $PADDLE_ROOT/tools/no_parallel_case_file
        noparallel_ut_endTime_s=`date +%s`
        echo "ipipe_log_param_noparallel_TestCases_Total_Time: $[ $noparallel_ut_endTime_s - $noparallel_ut_startTime_s ]s"
        echo "ipipe_log_param_noparallel_TestCases_Total_Time: $[ $noparallel_ut_endTime_s - $noparallel_ut_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
        ###retry
        collect_failed_tests
        rm -f $tmp_dir/*
        exec_times=0
        retry_unittests_record=''
        retry_time=4
        exec_time_array=('first' 'second' 'third' 'fourth')
        parallel_failed_tests_exec_retry_threshold=120
        exec_retry_threshold=30
        is_retry_execute=0
        rerun_ut_startTime_s=`date +%s`
        if [ -n "$failed_test_lists" ];then
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
                        for line in ${retry_unittests[@]} ;
                            do
                                if [[ "$retry_cases" == "" ]]; then
                                    retry_cases="^$line$"
                                else
                                    retry_cases="$retry_cases|^$line$"
                                fi
                            done

                        if [[ "$retry_cases" != "" ]]; then
			    # re-run test run 1 job
			    export CTEST_PARALLEL_LEVEL=1
                            card_test "$retry_cases" -1 1
                        fi
                        exec_times=$[$exec_times+1]
                        failed_test_lists=''
                        collect_failed_tests
                        rm -f $tmp_dir/*
                        retry_cases=''
                    else
                        break
                    fi
                done
            retry_unittests_record="$retry_unittests_record$failed_test_lists"
        fi
        rerun_ut_endTime_s=`date +%s`
        echo "ipipe_log_param_Rerun_TestCases_Total_Time: $[ $rerun_ut_endTime_s - $rerun_ut_startTime_s ]s"
        echo "ipipe_log_param_Rerun_TestCases_Total_Time: $[ $rerun_ut_endTime_s - $rerun_ut_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
        if [ "$WITH_ROCM" != "ON" ];then
            mkdir -p /home/data/cfs/coverage/${PR_ID}/${COMMIT_ID}/
            cp $PADDLE_ROOT/build/Testing/Temporary/CTestCostData.txt /home/data/cfs/coverage/${PR_ID}/${COMMIT_ID}/
        fi
        if [[ "$EXIT_CODE" != "0" ]]; then
            show_ut_retry_result
        fi
set -ex
    fi
}

function check_coverage() {
    if [ ${WITH_COVERAGE:-ON} == "ON" ] ; then
        /bin/bash ${PADDLE_ROOT}/ci/coverage_info.sh
    else
        echo "WARNING: check_coverage need to compile with WITH_COVERAGE=ON, but got WITH_COVERAGE=OFF"
    fi
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
    bash test.sh
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

function generate_api_spec() {
    set -e
    spec_kind=$2
    if [ "$spec_kind" != "PR" ] && [ "$spec_kind" != "DEV" ]; then
        echo "Not supported $2"
        exit 1
    fi
    if [ "$spec_kind" == "DEV" ]; then
        git checkout $BRANCH
    fi
    REQUIREMENTS_PATH=${PADDLE_ROOT}/python/requirements.txt
    PRINT_SIGNATURES_SCRIPT_PATH=${PADDLE_ROOT}/tools/print_signatures.py

    mkdir -p ${PADDLE_ROOT}/build/.check_api_workspace
    cd ${PADDLE_ROOT}/build/.check_api_workspace
    virtualenv -p `which python` .${spec_kind}_env
    source .${spec_kind}_env/bin/activate
    pip install -r $REQUIREMENTS_PATH

    if [ -d "${PADDLE_ROOT}/build/python/dist/" ]; then
        pip install ${PADDLE_ROOT}/build/python/dist/*whl
    elif [ -d "${PADDLE_ROOT}/dist/" ];then
        pip install ${PADDLE_ROOT}/dist/*whl
        mkdir ${PADDLE_ROOT}/build/python/dist/ && mv  ${PADDLE_ROOT}/dist/*whl  ${PADDLE_ROOT}/build/python/dist/
    fi
    spec_path=${PADDLE_ROOT}/paddle/fluid/API_${spec_kind}.spec
    python ${PRINT_SIGNATURES_SCRIPT_PATH} paddle > $spec_path
    python ${PRINT_SIGNATURES_SCRIPT_PATH} --show-fields="args,varargs,varkw,defaults,kwonlyargs,kwonlydefaults" paddle > ${spec_path}.api
    python ${PRINT_SIGNATURES_SCRIPT_PATH} --show-fields="annotations" paddle > ${spec_path}.annotations
    python ${PRINT_SIGNATURES_SCRIPT_PATH} --show-fields="document" paddle > ${spec_path}.doc

    # used to log op_register data_type
    op_type_path=${PADDLE_ROOT}/paddle/fluid/OP_TYPE_${spec_kind}.spec
    python ${PADDLE_ROOT}/tools/check_op_register_type.py > $op_type_path

    # used to log op_register data_type
    op_type_path=${PADDLE_ROOT}/paddle/fluid/OP_KERNEL_DTYPE_${spec_kind}.spec
    python ${PADDLE_ROOT}/tools/check_op_kernel_same_dtypes.py > $op_type_path

    # print all ops desc in dict to op_desc_path
    op_desc_path=${PADDLE_ROOT}/paddle/fluid/OP_DESC_${spec_kind}.spec
    python ${PADDLE_ROOT}/tools/print_op_desc.py > $op_desc_path

    # print api and the md5 of source code of the api.
    api_source_md5_path=${PADDLE_ROOT}/paddle/fluid/API_${spec_kind}.source.md5
    python ${PADDLE_ROOT}/tools/count_api_without_core_ops.py -p paddle > $api_source_md5_path

    python ${PADDLE_ROOT}/tools/diff_use_default_grad_op_maker.py \
        ${PADDLE_ROOT}/paddle/fluid/op_use_default_grad_maker_${spec_kind}.spec

    deactivate

    cd ${PADDLE_ROOT}/build
    rm -rf ${PADDLE_ROOT}/build/.check_api_workspace
    git checkout test
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

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

unset GREP_OPTIONS
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/utils.sh
init

# origin name: example
function exec_samplecode_test() {
    if [ -d "${PADDLE_ROOT}/build/pr_whl" ];then
        pip install ${PADDLE_ROOT}/build/pr_whl/*.whl
    else
        echo "WARNING: PR wheel is not found. Use develop wheel !!!"
        pip install ${PADDLE_ROOT}/build/python/dist/*.whl
    fi

    python -c "import paddle;print(paddle.__version__);paddle.version.show()"

    cd ${PADDLE_ROOT}/tools
    if [ "$1" = "cpu" ] ; then
        python sampcd_processor.py --mode cpu; example_error=$?
    elif [ "$1" = "gpu" ] ; then
        export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
        SAMPLE_CODE_EXEC_THREADS=${SAMPLE_CODE_EXEC_THREADS:-2}
        python sampcd_processor.py --threads=${SAMPLE_CODE_EXEC_THREADS} --mode gpu; example_error=$?
    fi
    if [ "$example_error" != "0" ];then
        echo "Code instance execution failed" >&2
        exit 5
    fi
}

function exec_type_checking() {
    if [ -d "${PADDLE_ROOT}/build/pr_whl" ];then
        pip install ${PADDLE_ROOT}/build/pr_whl/*.whl
    else
        echo "WARNING: PR wheel is not found. Use develop wheel !!!"
        pip install ${PADDLE_ROOT}/build/python/dist/*.whl
    fi

    python -c "import paddle;print(paddle.__version__);paddle.version.show()"

    cd ${PADDLE_ROOT}/tools

    # check all sample code
    TITLE_CHECK_ALL=`curl -s https://github.com/PaddlePaddle/Paddle/pull/${GIT_PR_ID} | grep "<title>" | grep -i "\[typing\]" || true`
    DEBUG_MODE=`curl -s https://github.com/PaddlePaddle/Paddle/pull/${GIT_PR_ID} | grep "<title>" | grep -i "\[debug\]" || true`

    if [[ ${TITLE_CHECK_ALL} ]]; then
        if [[ ${DEBUG_MODE} ]]; then
            python type_checking.py --debug --full-test; type_checking_error=$?
        else
            python type_checking.py --full-test; type_checking_error=$?
        fi
    else
        if [[ ${DEBUG_MODE} ]]; then
            python type_checking.py --debug; type_checking_error=$?
        else
            python type_checking.py; type_checking_error=$?
        fi
    fi

    if [ "$type_checking_error" != "0" ];then
        echo "Example code type checking failed" >&2
        exit 5
    fi
}

function summary_check_example_code_problems() {
    set +x
    local example_code=$1
    local example_info=$2

    if [ $example_code -ne 0 ];then
        echo "==============================================================================="
        echo "*****Example code error***** Please fix the error listed in the information:"
        echo "==============================================================================="
        echo "$example_info"
        echo "==============================================================================="
        echo "*****Example code FAIL*****"
        echo "==============================================================================="
    else
        echo "==============================================================================="
        echo "*****Example code info*****"
        echo "==============================================================================="
        echo "$example_info"
        echo "==============================================================================="
        echo "*****Example code PASS*****"
        echo "==============================================================================="
    fi
    set -x
}

function summary_type_checking_problems() {
    set +x
    local type_checking_code=$1
    local type_checking_info=$2

    if [ $type_checking_code -ne 0 ];then
        echo "==============================================================================="
        echo "*****Example code type checking error***** Please fix the error listed in the information:"
        echo "==============================================================================="
        echo "$type_checking_info"
        echo "==============================================================================="
        echo "*****Example code type checking FAIL*****"
        echo "==============================================================================="
    else
        echo "==============================================================================="
        echo "*****Example code type checking info*****"
        echo "==============================================================================="
        echo "$type_checking_info"
        echo "==============================================================================="
        echo "*****Example code type checking PASS*****"
        echo "==============================================================================="
    fi
    set -x
}

function exec_samplecode_checking() {
    # check sample code with doctest
    example_info_gpu=""
    example_code_gpu=0
    if [ "${WITH_GPU}" == "ON" ] ; then
        { example_info_gpu=$(exec_samplecode_test gpu 2>&1 1>&3 3>/dev/null); } 3>&1
        example_code_gpu=$?
    fi
    { example_info=$(exec_samplecode_test cpu 2>&1 1>&3 3>/dev/null); } 3>&1
    example_code=$?

    # check sample typing with mypy
    { type_checking_info=$(exec_type_checking 2>&1 1>&3 3>/dev/null); } 3>&1
    type_checking_code=$?

    # summary
    summary_check_example_code_problems $[${example_code_gpu} + ${example_code}] "${example_info_gpu}\n${example_info}"
    summary_type_checking_problems $type_checking_code "$type_checking_info"

    # exit with error code
    if [ $example_code -ne 0 ];then
        exit $example_code
    fi
    if [ $example_code_gpu -ne 0 ];then
        exit $example_code_gpu
    fi
    if [ $type_checking_code -ne 0 ];then
        exit $type_checking_code
    fi
}

export PATH=/usr/local/python3.10.0/bin:/usr/local/python3.10.0/include:/usr/local/bin:${PATH}
echo "export PATH=${PATH}" >> ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/compat:$LD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> ~/.bashrc
ln -sf $(which python${PY_VERSION}) /usr/local/bin/python
ln -sf $(which python${PY_VERSION}) /usr/bin/python
ln -sf $(which pip${PY_VERSION}) /usr/local/bin/pip
mkdir -p /home/data/cfs/.ccache/static-check

pip config set global.cache-dir "/home/data/cfs/.cache/pip"
pip install --upgrade pip 1>nul
pip install -r "${work_dir}/python/requirements.txt" 1>nul
pip install -r "${work_dir}/python/unittest_py/requirements.txt" 1>nul

exec_samplecode_checking

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

function parallel_test_base_gpups() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit GpuPS tests ...
    ========================================
EOF
        ut_startTime_s=`date +%s`
        set +e
        bash ${PADDLE_ROOT}/tools/gpups_test.sh
        EXIT_CODE=$?
        set -e
        ut_endTime_s=`date +%s`
        echo "GPUPS testCase Time: $[ $ut_endTime_s - $ut_startTime_s ]s"

        if [[ "$EXIT_CODE" != "0" ]]; then
            exit 8;
        fi
    fi
}

function parallel_fa_unit() {
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running FA unit tests in parallel way ...
    ========================================
EOF
    fi
    local parallel_list
    parallel_list="^test_block_multihead_attention$|\
    ^test_fused_weight_only_linear_pass$|\
    ^test_block_multihead_attention_gqa$|\
    ^test_fused_flash_attn_pass$|\
    ^test_fused_multi_transformer_op$|\
    ^test_fused_multi_transformer_int8_op$|\
    ^test_flash_attention$|\
    ^test_flash_attention_deterministic$|\
    ^test_flashmask$|\
    ^test_fused_gate_attention_op$"
    get_quickly_disable_ut||disable_ut_quickly='disable_ut'

    card_test "${parallel_list}" 1

    collect_failed_tests
    rm -f $tmp_dir/*
    if [ -n "$failed_test_lists" ];then
        echo "Sorry, some FA tests failed."
        collect_failed_tests
        echo "Summary Failed Tests... "
        echo "========================================"
        echo "The following tests FAILED: "
        echo "${failed_test_lists}"| sort -u
        exit 8
    fi
}

function distribute_test() {
    python ${PADDLE_ROOT}/tools/get_pr_title.py skip_distribute_test && CINN_OR_BUAA_PR=1
    if [[ "${CINN_OR_BUAA_PR}" = "1" ]];then
        echo "PR's title with 'CINN' or 'BUAA', skip the run distribute ci test !"
        exit 0
    fi
    echo "::group::Start gpups tests"
    parallel_test_base_gpups
    echo "End gpups tests"
    echo "::endgroup::"

    echo "::group::Start FA tests"
    parallel_fa_unit
    echo "End FA tests"
    echo "::endgroup::"

    echo "Downloading PaddleNLP_stable_paddle.tar.gz..."
    cd ${work_dir}
    wget https://paddlenlp.bj.bcebos.com/wheels/PaddleNLP_stable_paddle.tar.gz --no-proxy
    echo "Extracting PaddleNLP_stable_paddle.tar.gz..."
    tar -zxf PaddleNLP_stable_paddle.tar.gz
    cd PaddleNLP
    sed -i '/lac/d' scripts/regression/requirements_ci.txt

    echo "::group::Install paddlenlp dependencies"
    pip install -r requirements.txt
    pip install -r scripts/regression/requirements_ci.txt
    pip install -r ./csrc/requirements.txt
    echo "::endgroup::"

    echo "::group::Install PaddleNLP"
    python setup.py install
    python -m pip install pytest-timeout
    cd csrc && python setup_cuda.py install
    echo "::endgroup::"

    cd ${work_dir}
    wget -q --no-proxy https://paddle-qa.bj.bcebos.com/paddlenlp/Bos.zip --no-check-certificate
    unzip -P'41maLgwWnCLaFTCehlwQ6n4l3oZpS/r5gPq4K4VLj5M1024' Bos.zip
    mkdir paddlenlp && mv Bos/* ./paddlenlp/
    rm -rf ./paddlenlp/upload/*
    rm -rf ./paddlenlp/models/bigscience/*

    echo "::group::Start LLM Test"
    cd ${work_dir}/PaddleNLP
    # Disable Test: test_gradio
    rm tests/llm/test_gradio.py
    # python -m pytest -s -v tests/llm --timeout=3600
    echo "End LLM Test"
    echo "::endgroup::"

    echo "::group::Start auto_parallel Test"
    cd ${work_dir}
    timeout 50m bash ci/auto_parallel/ci_distributed_stable.sh
    EXIT_CODE=$?
    echo "End auto_parallel Test"
    echo "::endgroup::"

    if [[ "$EXIT_CODE" != "0" ]]; then
      exit 8;
    fi
}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
PATH=/usr/local/bin:${PATH}
ln -sf $(which python3.10) /usr/local/bin/python
ln -sf $(which pip3.10) /usr/local/bin/pip
echo "::group::Install paddle dependencies"
pip config set global.cache-dir "/root/.cache/pip"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# node(swgu98): Switching to the distribute2 machine will install 1.18, which is inconsistent with the previous behavior
pip install onnx==1.17.0
pip install -r "${work_dir}/python/requirements.txt"
pip install -r "${work_dir}/python/unittest_py/requirements.txt"
pip install ${work_dir}/dist/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl
echo "::endgroup::"
ldconfig

distribute_test

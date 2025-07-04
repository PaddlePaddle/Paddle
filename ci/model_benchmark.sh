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

export PATH=$(pwd)/run_env:${PATH}
push_file=${ROOT_DIR}/PaddleTest/tools/bos_tools.py

set_python () {
    cd ${ROOT_DIR}
    set +e
    rm -rf run_env
    mkdir run_env
    ln -s $(which python3.10) run_env/python
    ln -s $(which pip3.10) run_env/pip
    pip install --upgrade pip
    pip config set global.cache-dir "/home/data/cfs/.cache/pip"
    echo "----------benchmark_log: python env done; pip list ------------- "
    pip list
}

wget_model_benchmark_ci () {
    cd ${ROOT_DIR}
    wget --no-proxy https://paddle-github-action.bj.bcebos.com/paddle-qa/benchmark/${bos_path}/model_benchmark_ci_action.tar.gz
    tar xvf model_benchmark_ci_action.tar.gz
    mv model_benchmark_ci_action/* ${ROOT_DIR}

    rm -rf benchmark_summary_res.csv
    wget --no-proxy https://paddle-github-action.bj.bcebos.com/paddle-qa/benchmark/${bos_path}/benchmark_summary_res.csv
    cd ${ROOT_DIR}
    wget -q --no-proxy https://xly-devops.bj.bcebos.com/PaddleTest/PaddleTest.tar.gz
    tar xf PaddleTest.tar.gz
    ls
}

get_paddlescope_tar () {
    cd ${ROOT_DIR}
    cp ${CACHE_DIR}/paddlescope_nosql.tar.gz .
    tar xf paddlescope_nosql.tar.gz
    pip install -r paddlescope/requirements.txt
    echo "----------benchmark_log: paddlescope python env done; pip list ------------- "
    pip list;
}

update_summary () {
    cd ${ROOT_DIR}
    rm -rf benchmark_summary_res.csv
    wget --no-proxy https://paddle-github-action.bj.bcebos.com/paddle-qa/benchmark/${bos_path}/benchmark_summary_res.csv
    bash main.sh update_summary
    python ${push_file} benchmark_summary_res.csv paddle-github-action/paddle-qa/benchmark/${bos_path} upload
    echo "----------benchmark_log: update_summary done ------------- "
    head -3 benchmark_summary_res.csv
}

check_paddle () {
    set_python
    cd ${ROOT_DIR}/paddle
    git checkout test
    git diff --numstat --diff-filter=AMR ${BRANCH} | \
    grep -v inference/ | grep -v test/ | grep -v tools/ | \
    grep -v onednn/ | grep -v security/ | grep -v doc/ | \
    grep -v xpu/ | \
    grep -v '\.md$' | grep -v '\.sh$' | \
    awk '{print $NF}' | tee filelist

    git diff --numstat --diff-filter=AMR ${BRANCH} | \
    grep -v inference/ | grep -v test/ | grep -v tools/ | \
    grep -v onednn/ | grep -v security/ | grep -v doc/ | \
    grep -v xpu/ | \
    grep -v '\.md$' | grep -v '\.sh$' | \
    grep operators/ | awk '{print $NF}' | tee operatorlist

    filelist_num=`cat filelist | wc -l`
    echo "filelist_num: ${filelist_num}"
    if [ ${filelist_num} -eq 0 ];then
        echo "The modified files does not affect models in PR-CI-Model-benchmark, so skip this ci."
        update_summary
        echo "can-skip=true" >> $1
        exit 0
    fi
    # check cinn
    git diff --numstat --diff-filter=AMRD ${BRANCH} | grep paddle/cinn | awk '{print $NF}' | tee pr_filelist1.log
    filelist_num1=`cat pr_filelist1.log | wc -l`

    git diff --numstat --diff-filter=AMRD ${BRANCH} | grep paddle/fluid/primitive/ | awk '{print $NF}' | tee pr_filelist2.log
    filelist_num2=`cat pr_filelist2.log | wc -l`

    git diff --numstat --diff-filter=AMRD ${BRANCH} | grep paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/ | awk '{print $NF}' | tee pr_filelist3.log
    filelist_num3=`cat pr_filelist3.log | wc -l`

    git diff --numstat --diff-filter=AMRD ${BRANCH} | grep paddle/fluid/pir/dialect/operator/interface | awk '{print $NF}' | tee pr_filelist4.log
    filelist_num4=`cat pr_filelist4.log | wc -l`

    sum_num=$((filelist_num1 + filelist_num2 + filelist_num3 + filelist_num4))
    echo " check cinn case sum_num: ${sum_num}"
    export CINN_FILE_LIST=${sum_num}
    echo "export CINN_FILE_LIST=$CINN_FILE_LIST" >> ~/.bashrc
}

check_skip () {
    cd ${ROOT_DIR}
    export IF_RUN='False'
    bash main.sh get_case > get_case.log 2>&1
    echo "----------benchmark_log: get_case.log "
    cat get_case.log
    echo "----------benchmark_log: get_case.log done "
    IF_RUN=`cat get_case.log |grep benchmark_log: |grep IF_RUN |awk -F ' ' '{print $3}'`
    echo "----------benchmark_log: IF_RUN, ${IF_RUN} ------------- "
    if [[ ${IF_RUN} == 'False' ]];then
        echo "----------benchmark_log: skip model_benchmark ------------- "
        update_summary
        echo "can-skip=true" >> $1
        exit 0
    fi
    if [ -f "temp_summary_res.txt" ]; then
        cat temp_summary_res.txt
    else
        echo "temp_summary_res.txt not exist"
    fi
    cd ${ROOT_DIR}
    python ${push_file} benchmark_summary_res.csv paddle-github-action/paddle-qa/benchmark/${bos_path} upload
    cat $ROOT_DIR/case_info.csv
}
check_model_benchmark () {
    # benchmark case
    cd ${ROOT_DIR}
    bash main.sh run_all;EXCODE=$?
    # over
    cd ${ROOT_DIR}
    bash main.sh update_history
    python ${push_file} benchmark_history_PR_result.csv paddle-github-action/paddle-qa/benchmark/${bos_path} upload
    update_summary
    echo ------------------------------ model_benchmark CI result   ------------------------------
    cat $ROOT_DIR/case_info.csv
}

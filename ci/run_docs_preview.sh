# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

export BRANCH=${AGILE_COMPILE_BRANCH}
export GIT_PR_ID=${AGILE_PULL_ID}

export no_proxy=mirrors.tuna.tsinghua.edu.cn,bcebos.com,baidu.com,mirror.baidu.com,baidu-int.com,paddlepaddle.org.cn,localhost,127.0.0.1
git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"

set +e

if [ -f "/home/data/cfs/api_doc/${AGILE_PULL_ID}/api_doc_break.flag" ]; then
    echo 'API documents no change, skip doc build.'
    exit 0
fi

DOCS_REPO=https://github.com/PaddlePaddle/docs.git

export FLUIDDOCDIR=/FluidDoc
if [ ! -d ${FLUIDDOCDIR} ] ; then
    DOCS_REPO_DAILY="https://github-repo-tgz.bj.bcebos.com/PaddlePaddle/docs/${BRANCH}.tgz"
    echo "DOCS_REPO_DAILY is ${DOCS_REPO_DAILY}"
    TMP_TGZ=/tmp/docs-repo.tgz
    http_code=$(curl -sL -w "%{http_code}" -o ${TMP_TGZ} -X GET -k ${DOCS_REPO_DAILY})
    if [ "${http_code}" = "200" ] ; then
        mkdir -p ${FLUIDDOCDIR}
        tar xzf ${TMP_TGZ} -C ${FLUIDDOCDIR}
        cd ${FLUIDDOCDIR}
        pwd
        ls
        # git remote add upstream ${DOCS_REPO}
        git fetch upstream $BRANCH
    else
        echo "curl -I ${DOCS_REPO_DAILY} got http_code=${http_code}"
        git clone --origin upstream --depth=200 --branch=${BRANCH} ${DOCS_REPO} ${FLUIDDOCDIR}
        if [ $? -ne 0 ];then
            exit 1
        fi
    fi
    cd ${FLUIDDOCDIR}
    pwd
    ls
else
    cd ${FLUIDDOCDIR}
    git reset --hard
    git clean -dfx
fi


# clone paddle repo
PADDLE_REPO=https://github.com/PaddlePaddle/Paddle.git

export PADDLE_DIR=/Paddle

if [ ! -d ${PADDLE_DIR} ] ; then
    git clone --origin upstream --depth=200 --branch=${BRANCH} ${PADDLE_REPO} ${PADDLE_DIR}
    if [ $? -ne 0 ];then
        exit 1
    fi
    cd ${PADDLE_DIR}
    git fetch upstream ${AGILE_COMPILE_BRANCH_REF}
    if [ $? -ne 0 ];then
        exit 1
    fi
    git checkout -b origin_pr FETCH_HEAD
    git --no-pager log --pretty=oneline -10
fi

cd ${FLUIDDOCDIR}/ci_scripts
ls
bash ci_start_en.sh

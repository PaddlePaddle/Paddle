#!/bin/bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

is_shell_attribute_set() { # attribute, like "x"
  case "$-" in
    *"$1"*) return 0 ;;
    *)    return 1 ;;
  esac
}
function get_docs_pr_num_from_paddle_pr_info(){
    # get_repo_pr_info's output
    pr_info_file=$1
    if [ ! -r ${pr_info_file} ] ; then
        return 1
    fi

    declare -A arr_kv
    while read line
    do
        echo "$line" | grep '^\w\+\s*=\s*.*' > /dev/null
        if [ $? = 0 ] ; then
            kv=($(echo $line | sed 's/=/\n/g'))
            k=($(echo "${kv[0]}" | sed 's/\s//g'))
            v=($(echo "${kv[1]}" | sed 's/^\s*//g' | sed 's/\s*$//g'))
            # arr_kv[${kv[1]}]=${kv[2]}
            arr_kv[${k}]=${v}
        fi
    done < <(jq -r '.body' ${pr_info_file})

    echo ${arr_kv[PADDLEDOCS_PR]}
    return 0
}

# Attention:
# 1. /FluidDoc will be used as the workspace of PaddlePaddle/docs. 
# 2. And /docs is used as the output of doc-build process.
# 3. If conflicted with yours, please modify the defination of FLUIDDOCDIR and
#    OUTPUTDIR in the subsequent codes.
# 4. The doc-build process is controlled under EnvVar BUILD_DOC and UPLOAD_DOC.
#    All the Chinese and English docs will be generated, and then uploaded.

PREVIEW_URL_PROMPT="ipipe_log_param_preview_url: None"
BUILD_DOC=${BUILD_DOC:=false}
UPLOAD_DOC=${UPLOAD_DOC:=false}

CURPWD=${PWD}

if [ -f /usr/local/python3.7.0/bin/sphinx-build ] ; then
    if [ -f /usr/local/bin/sphinx-build ] ; then
        rm /usr/local/bin/sphinx-build
    fi
    ln -s /usr/local/python3.7.0/bin/sphinx-build /usr/local/bin/sphinx-build
fi

if [ "${BUILD_DOC}" = "true" ] &&  [ -x /usr/local/bin/sphinx-build ] ; then
    export FLUIDDOCDIR=${FLUIDDOCDIR:=/FluidDoc}
    export OUTPUTDIR=${OUTPUTDIR:=/docs}
    export VERSIONSTR=$(echo ${BRANCH} | sed 's@release/@@g')

    if [ -d ${FLUIDDOCDIR} ] ; then
        echo "${FLUIDDOCDIR} exists, git clone will be skipped, but git clean will be done."
        cd ${FLUIDDOCDIR}
        git reset --hard
        git clean -dfx
        cd ${CURPWD}
    else
        git clone -b ${BRANCH} --depth=1 https://github.com/PaddlePaddle/docs.git ${FLUIDDOCDIR}
        if [ ! "$?" = "0" ] ; then
            git clone --depth=1 https://github.com/PaddlePaddle/docs.git ${FLUIDDOCDIR}
        fi
    fi
    if [ -d ${OUTPUTDIR} ] ; then
        echo "$0: rm -rf ${OUTPUTDIR}"
        rm -rf ${OUTPUTDIR}
        mkdir -p ${OUTPUTDIR}
    fi

    # install requirements
    export no_proxy=mirror.baidu.com,${no_proxy}
    apt-get install -y --no-install-recommends doxygen jq
    echo 'beautifulsoup4
Markdown
sphinx-sitemap
sphinx-markdown-tables
breathe
exhale
sphinx_design
nbsphinx
' >/tmp/doc-build.requirements && \
    pip install --no-cache-dir -i https://mirror.baidu.com/pypi/simple -r /tmp/doc-build.requirements && \
    rm /tmp/doc-build.requirements


    source ${FLUIDDOCDIR}/ci_scripts/utils.sh
    paddle_pr_info=$(get_repo_pr_info "PaddlePaddle/Paddle" ${GIT_PR_ID})
    docs_pr_id=$(get_docs_pr_num_from_paddle_pr_info ${paddle_pr_info})
    if [ -n "${docs_pr_id}" ] ; then
        cd ${FLUIDDOCDIR}
        git fetch --depth=1 origin pull/${docs_pr_id}/head
        git checkout -b "pr${docs_pr_id}" FETCH_HEAD
        git log --pretty=oneline -10
    fi
    echo "docs_pr_id=${docs_pr_id}"


    # build doc
    /bin/bash -x ${FLUIDDOCDIR}/ci_scripts/gendoc.sh
    if [ $? -ne 0 ];then
        echo 'gendoc error'
        exit 1
    fi

    if [ "${UPLOAD_DOC}" = "true" ] ; then
        curl -o /tmp/linux-bcecmd-0.3.0.zip https://sdk.bce.baidu.com/console-sdk/linux-bcecmd-0.3.0.zip && \
        python -m zipfile -e /tmp/linux-bcecmd-0.3.0.zip /opt && \
        chmod +x /opt/linux-bcecmd-0.3.0/bcecmd && \
        rm /tmp/linux-bcecmd-0.3.0.zip && \
        curl -o /tmp/boscmdconfig.tgz https://paddle-dev-tools-open.bj.bcebos.com/fluiddoc-preview/boscmdconfig.tgz && \
        tar xzf /tmp/boscmdconfig.tgz -C /opt/linux-bcecmd-0.3.0/ && \
        rm /tmp/boscmdconfig.tgz

        # credentials file is empty, please build it if need.
        BCECMD=/opt/linux-bcecmd-0.3.0/bcecmd
        BCECMD_CONFIG=/opt/linux-bcecmd-0.3.0/boscmdconfig

        is_shell_attribute_set x
        xdebug_setted=$?
        if [ $xdebug_setted ] ; then
            set +x
        fi
        if [ -n "${BOS_CREDENTIAL_AK}" ] && [ -n "${BOS_CREDENTIAL_SK}" ] ; then
            echo "Ak = ${BOS_CREDENTIAL_AK}" >> ${BCECMD_CONFIG}/credentials
            echo "Sk = ${BOS_CREDENTIAL_SK}" >> ${BCECMD_CONFIG}/credentials
        fi
        if [ $xdebug_setted ] ; then
            set -x
        fi

        PREVIEW_JOB_NAME="preview-paddle-pr-${GIT_PR_ID}"
        BOSBUCKET=${BOSBUCKET:=paddle-site-web-dev}
        ${BCECMD} --conf-path ${BCECMD_CONFIG} bos sync "${OUTPUTDIR}/en/${VERSIONSTR}" "bos:/${BOSBUCKET}/documentation/en/${PREVIEW_JOB_NAME}" \
            --delete --yes --exclude "${OUTPUTDIR}/en/${VERSIONSTR}/_sources/"
        ${BCECMD} --conf-path ${BCECMD_CONFIG} bos sync "${OUTPUTDIR}/en/${VERSIONSTR}" "bos:/${BOSBUCKET}/documentation/en/${PREVIEW_JOB_NAME}" \
            --delete --yes --exclude "${OUTPUTDIR}/en/${VERSIONSTR}/_sources/"
        ${BCECMD} --conf-path ${BCECMD_CONFIG} bos sync "${OUTPUTDIR}/zh/${VERSIONSTR}" "bos:/${BOSBUCKET}/documentation/zh/${PREVIEW_JOB_NAME}" \
            --delete --yes --exclude "${OUTPUTDIR}/zh/${VERSIONSTR}/_sources/"
        ${BCECMD} --conf-path ${BCECMD_CONFIG} bos sync "${OUTPUTDIR}/zh/${VERSIONSTR}" "bos:/${BOSBUCKET}/documentation/zh/${PREVIEW_JOB_NAME}" \
            --delete --yes --exclude "${OUTPUTDIR}/zh/${VERSIONSTR}/_sources/"
        PREVIEW_URL_PROMPT="ipipe_log_param_preview_url: http://${PREVIEW_JOB_NAME}.${PREVIEW_SITE:-paddle.run}/documentation/docs/zh/api/index_cn.html"
    fi
fi

cd ${CURPWD}
# print the preview url
echo "${PREVIEW_URL_PROMPT}"

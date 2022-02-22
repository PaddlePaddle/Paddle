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

# Attention: this script will rm folder /FluidDoc and /docs first, then reuse the two folders.

# build all the Chinese and English docs, and upload them. Controlled with Env BUILD_DOC and UPLOAD_DOC
PREVIEW_URL_PROMPT="ipipe_log_param_preview_url: None"
BUILD_DOC=${BUILD_DOC:=true}
UPLOAD_DOC=${UPLOAD_DOC:=false}

CURPWD=${PWD}

if [ ! -f /usr/local/bin/sphinx-build ] && [ -f /usr/local/python3.7.0/bin/sphinx-build ] ; then
    ln -s /usr/local/python3.7.0/bin/sphinx-build /usr/local/bin/sphinx-build
fi

if [ "${BUILD_DOC}" = "true" ] &&  [ -x /usr/local/bin/sphinx-build ] ; then
    export FLUIDDOCDIR=/FluidDoc
    export OUTPUTDIR=/docs
    export VERSIONSTR=$(echo ${BRANCH} | sed 's@release/@@g')

    if [ -d ${FLUIDDOCDIR} ] ; then
        # echo "$0: rm -rf ${FLUIDDOCDIR}"
        # rm -rf ${FLUIDDOCDIR}
        echo "${FLUIDDOCDIR} exists, git clone skipped."
    else
        git clone --depth=1 https://github.com/PaddlePaddle/docs.git ${FLUIDDOCDIR}
        # TODO: checkout the required docs PR?
    fi
    if [ -d ${OUTPUTDIR} ] ; then
        echo "$0: rm -rf ${OUTPUTDIR}"
        rm -rf ${OUTPUTDIR}
        mkdir -p ${OUTPUTDIR}
    fi
    # install requirements
    mkdir -p /config
    curl -L -o /config/shpinx-docs-config.tgz https://paddle-dev-tools-open.bj.bcebos.com/fluiddoc-preview/shpinx-docs-config.tgz
    tar xzf /config/shpinx-docs-config.tgz -C /config
    curl -L -o /root/post_filter_htmls.py https://paddle-dev-tools-open.bj.bcebos.com/fluiddoc-preview/post_filter_htmls.py

    apt-get update && apt-get install -y --no-install-recommends doxygen
    # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 
    pip install beautifulsoup4
    pip install Markdown
    pip install sphinx-sitemap
    pip install sphinx-markdown-tables
    pip install breathe
    pip install exhale
    pip install sphinx_design
    pip install nbsphinx
    # exhale 和 breache 这两个真实需要么？

    # build doc
    /bin/bash -x ${FLUIDDOCDIR}/ci_scripts/gendoc.sh
    if [ $? -ne 0 ];then
        echo 'gendoc error'
        exit 1
    fi

    if [ "${UPLOAD_DOC}" = "true" ] ; then
        BCECMD=
        BCECMD_CONFIG=

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
        # print preview url
        PREVIEW_URL_PROMPT="ipipe_log_param_preview_url: http://${PREVIEW_JOB_NAME}.${PREVIEW_SITE:-preview.paddlepaddle.org}/documentation/docs/zh/api/index_cn.html"
    fi
fi
cd ${CURPWD}
echo "${PREVIEW_URL_PROMPT}"

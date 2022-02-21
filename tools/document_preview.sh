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
        echo "$0: rm -rf ${FLUIDDOCDIR}"
        rm -rf ${FLUIDDOCDIR}
        git clone https://github.com/PaddlePaddle/docs.git ${FLUIDDOCDIR}
        # TODO: checkout the required docs PR?
    fi
    if [ -d ${OUTPUTDIR} ] ; then
        echo "$0: rm -rf ${OUTPUTDIR}"
        rm -rf ${OUTPUTDIR}
        mkdir -p ${OUTPUTDIR}
    fi
    # install requirements

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

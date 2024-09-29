#!/bin/bash

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


if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
failed_num=0
echo_list=()

function check_approval(){
    person_num=`echo $@|awk '{for (i=2;i<=NF;i++)print $i}'`
    APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py $1 $person_num`
    if [[ "${APPROVALS}" == "FALSE" && "${echo_line}" != "" ]]; then
        add_failed "${failed_num}. ${echo_line}"
    fi
}

function add_failed(){
    failed_num=`expr $failed_num + 1`
    echo_list="${echo_list[@]}$1"
}


api_params_diff=`python ${PADDLE_ROOT}/tools/check_api_compatible.py ${PADDLE_ROOT}/paddle/fluid/API_DEV.spec  ${PADDLE_ROOT}/paddle/fluid/API_PR.spec`
api_spec_diff=`python ${PADDLE_ROOT}/tools/diff_api.py ${PADDLE_ROOT}/paddle/fluid/API_DEV.spec.api  ${PADDLE_ROOT}/paddle/fluid/API_PR.spec.api`
api_annotation_diff=`python ${PADDLE_ROOT}/tools/diff_api.py ${PADDLE_ROOT}/paddle/fluid/API_DEV.spec.annotations  ${PADDLE_ROOT}/paddle/fluid/API_PR.spec.annotations`
if [ "$api_spec_diff" != "" -o "${api_params_diff}" != "" ]; then
    echo_line="You must have one RD (XiaoguangHu01, jeff41404 or qingqing01) approval for API change.\n"

    check_approval 1 XiaoguangHu01 jeff41404 qingqing01
fi

if [ "$api_annotation_diff" != "" ]; then
    echo_line="You must have one member of Typing group (SigureMo, megemini, zrr1999, sunzhongkai588, luotao1) approval for API annotation change.\n"
    check_approval 1 SigureMo megemini zrr1999 sunzhongkai588 luotao1
fi

api_yaml_diff=`python ${PADDLE_ROOT}/tools/check_api_yaml_same.py ${PADDLE_ROOT}/paddle/fluid/API_DEV.spec  ${PADDLE_ROOT}/paddle/fluid/API_PR.spec ${BRANCH} ${PADDLE_ROOT}`
if [ "$api_yaml_diff" != "" ]; then
    echo_line="API's name and params should be consistent with op's name and params in yaml.
                The API or Yaml file you changed may cause inconsistent.\n"
    echo_line="${echo_line} please request one of the RD (YuanRisheng, zyfncg, phlrain) review and approve.\n"
    echo_line="${echo_line}\r\n ${api_yaml_diff}\n"
    check_approval 1 YuanRisheng zyfncg phlrain
fi

op_type_spec_diff=`python ${PADDLE_ROOT}/tools/check_op_register_type.py ${PADDLE_ROOT}/paddle/fluid/OP_TYPE_DEV.spec  ${PADDLE_ROOT}/paddle/fluid/OP_TYPE_PR.spec`
if [ "$op_type_spec_diff" != "" ]; then
    echo_line="You must have one RD (Aurelius84 (Recommend) or zhhsplendid)approval for the data_type registration of new operator. More data_type of new operator should be registered in your PR. Please make sure that both float/double (or int/int64_t) have been registered.\n For more details, please click [https://github.com/PaddlePaddle/Paddle/wiki/Data-types-of-generic-Op-must-be-fully-registered].\n"
    check_approval 1 Aurelius84 zhhsplendid
fi

op_kernel_dtype_spec_diff=`python ${PADDLE_ROOT}/tools/check_op_kernel_same_dtypes.py ${PADDLE_ROOT}/paddle/fluid/OP_KERNEL_DTYPE_DEV.spec  ${PADDLE_ROOT}/paddle/fluid/OP_KERNEL_DTYPE_PR.spec`
if [ "$op_kernel_dtype_spec_diff" != "" ]; then
    echo_line="You have added or modified Op Kernel, resulting in inconsistent data types supported by the forward and backward kernels of the same op, such modifications are not allowed in principle. If it is a mismatch, please request one RD (Aurelius84 or zyfncg) review and approve. Including the following kernels:\n${op_kernel_dtype_spec_diff}\n"
    check_approval 1 Aurelius84 zyfncg
fi

op_desc_diff=`python ${PADDLE_ROOT}/tools/check_op_desc.py ${PADDLE_ROOT}/paddle/fluid/OP_DESC_DEV.spec  ${PADDLE_ROOT}/paddle/fluid/OP_DESC_PR.spec`
inference_approve=`echo "$op_desc_diff" | grep "need inference to review" -`
slim_approve=`echo "$op_desc_diff" | grep "need slim to review" -`
if [ "$op_desc_diff" != "" ]; then
    echo_line="You must have one RD (inference[ vivienfanghuagood(Recommend), yuanlehome, qingqing01 ] or slim[ wanghaoshuang(Recommend), qingqing01 ] or train[ phlrain ]) approval for the changes of Inputs/Output/Attrs of OPs. The changes of OPs will cause that the new version inference fails to load model trained by the old version. Please modify your code. \n For more details, please click [https://github.com/PaddlePaddle/Paddle/wiki/OP-Input-Output-Attribute-Compatibility-Modification].\n${op_desc_diff}\n"
    check_approval 1 vivienfanghuagood yuanlehome qingqing01 wanghaoshuang phlrain
fi

if [ "$slim_approve" != "" ]; then
    echo_line="You must have one RD (wanghaoshuang(Recommend), qingqing01) approval for the changes of `quant` Inputs/Output/Attrs of OPs. \n For more details, please click [https://github.com/PaddlePaddle/Paddle/wiki/OP-Input-Output-Attribute-Compatibility-Modification].\n${slim_approve}\n"
    check_approval 1 wanghaoshuang qingqing01
fi

if [ "$inference_approve" != "" ]; then
    echo_line="You must have one RD (qingqing01(Recommend), heavengate) approval for the changes of `def` Inputs/Output/Attrs of OPs. \n For more details, please click [https://github.com/PaddlePaddle/Paddle/wiki/OP-Input-Output-Attribute-Compatibility-Modification].\n${inference_approve}\n"
    check_approval 1 qingqing01 heavengate
fi


DEV_OP_USE_DEFAULT_GRAD_MAKER_SPEC=${PADDLE_ROOT}/paddle/fluid/op_use_default_grad_maker_DEV.spec
PR_OP_USE_DEFAULT_GRAD_MAKER_SPEC=${PADDLE_ROOT}/paddle/fluid/op_use_default_grad_maker_PR.spec
ADDED_OP_USE_DEFAULT_GRAD_MAKER=`python ${PADDLE_ROOT}/tools/diff_use_default_grad_op_maker.py ${DEV_OP_USE_DEFAULT_GRAD_MAKER_SPEC} ${PR_OP_USE_DEFAULT_GRAD_MAKER_SPEC}`
if [ "${ADDED_OP_USE_DEFAULT_GRAD_MAKER}" != "" ]; then
  echo_line="You must have one RD (zhiqiu (Recommend) or zhhsplendid) approval because you use DefaultGradOpMaker for ${ADDED_OP_USE_DEFAULT_GRAD_MAKER}, which manages the grad_op memory optimization.\n"
  check_approval 1 zhiqiu zhhsplendid
fi

OUTPUT_LOG=`git diff -U0 upstream/$BRANCH | grep "^+" | grep -Ew "print|printf|fprintf|std::cout" || true`
if [ "$OUTPUT_LOG" != "" ];then
    git diff -U0 upstream/$BRANCH |grep "^+" | grep -Ew "print|printf|fprintf|std::cout"|sed 's#[ ][ ]##g'|sed 's#+##g' >/tmp/print.txt
    samplecode=`find tools/samplecode_temp -type f || true`
    sample_status=0
    if [ "$samplecode" != "" ];then
        cat `find tools/samplecode_temp -type f` >/tmp/samplecode.txt
        sed -i s#\"#\'#g /tmp/samplecode.txt
        while read line
        do
            code_in=`grep "$line" /tmp/samplecode.txt || true`
            if [ "$code_in" == "" ];then
                sample_status=1
            fi
        done</tmp/print.txt
    fi

    if [ "$sample_status" == 1 ] || [ "$samplecode" == "" ] ;then
        echo_line="print or std::cout is not recommended for direct use, please use logging or VLOG. If it is necessary to use, please contact tianshuo78520a (Recommend) or zhangbo9674 or SigureMo review and approve.\n"
        check_approval 1 tianshuo78520a zhangbo9674 SigureMo
    fi
fi

if [ -n "${echo_list}" ];then
  echo "**************************************************************"
  echo "Please find RD for approval first, and then find TPM for approval."
  echo -e "${echo_list[@]}"
  echo "There are ${failed_num} approved errors."
  echo "**************************************************************"

  # L40 L48 L62 has fetch the result out, but there are splitted.
  if [ "${api_spec_diff}" != "" -o "${api_annotation_diff}" != "" ] ; then
    python ${PADDLE_ROOT}/tools/diff_api.py ${PADDLE_ROOT}/paddle/fluid/API_DEV.spec  ${PADDLE_ROOT}/paddle/fluid/API_PR.spec
  fi
  if [ "${api_params_diff}" != "" ] ; then
    echo "api_params_diff: ${api_params_diff}"
  fi
  if [ "${op_type_spec_diff}" != "" ] ; then
    echo "op_type_spec_diff: ${op_type_spec_diff}"
  fi
  exit 6
fi

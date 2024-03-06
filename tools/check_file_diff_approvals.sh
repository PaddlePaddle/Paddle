#!/bin/bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
API_FILES=("CMakeLists.txt"
           "paddle/fluid/framework/operator.h"
           "paddle/fluid/framework/tensor.h"
           "paddle/fluid/framework/details/op_registry.h"
           "paddle/fluid/framework/grad_op_desc_maker.h"
           "paddle/fluid/framework/lod_tensor.h"
           "paddle/fluid/framework/selected_rows.h"
           "paddle/fluid/framework/op_desc.h"
           "paddle/fluid/framework/block_desc.h"
           "paddle/fluid/framework/var_desc.h"
           "paddle/fluid/framework/scope.h"
           "paddle/fluid/framework/ir/node.h"
           "paddle/fluid/framework/ir/graph.h"
           "paddle/fluid/framework/framework.proto"
           "python/paddle/distributed/__init"
           "python/paddle/distributed/fleet/__init__.py"
           "python/paddle/distributed/fleet/launch.py"
           "python/requirements.txt"
           "python/paddle/base/__init__.py"
           "python/paddle/base/compiler.py"
           "python/paddle/base/parallel_executor.py"
           "python/paddle/base/framework.py"
           "python/paddle/base/backward.py"
           "paddle/fluid/operators/distributed/send_recv.proto.in"
           "paddle/fluid/framework/unused_var_check.cc"
           "python/paddle/base/tests/unittests/white_list/check_shape_white_list.py"
           "python/paddle/base/tests/unittests/white_list/op_accuracy_white_list.py"
           "python/paddle/base/tests/unittests/white_list/compile_vs_runtime_white_list.py"
           "python/paddle/base/tests/unittests/white_list/no_check_set_white_list.py"
           "python/paddle/base/tests/unittests/white_list/check_op_sequence_instance_0_input_white_list.py"
           "python/paddle/base/tests/unittests/white_list/op_threshold_white_list.py"
           "python/paddle/base/tests/unittests/white_list/check_op_sequence_batch_1_input_white_list.py"
           "python/paddle/base/tests/unittests/white_list/no_grad_set_white_list.py"
           "tools/print_signatures.py"
           "tools/sampcd_processor.py"
           "tools/check_pr_approval.py"
           "paddle/scripts/paddle_build.bat"
           "tools/windows/run_unittests.sh"
           "tools/parallel_UT_rule.py"
           "python/paddle/base/dygraph/layers.py"
           "paddle/fluid/eager/grad_node_info.h"
           "paddle/fluid/eager/grad_node_info.cc"
           "paddle/fluid/eager/grad_tensor_holder.h"
           "paddle/fluid/eager/grad_tensor_holder.cc"
           "paddle/fluid/eager/tensor_wrapper.h"
           "paddle/fluid/eager/autograd_meta.cc"
           "paddle/fluid/eager/autograd_meta.h"
           "paddle/fluid/eager/backward.cc"
           "paddle/fluid/eager/backward.h"
           "paddle/phi/api/include/tensor.h"
           "paddle/phi/core/tensor_base.h"
           "paddle/phi/core/dense_tensor.h"
           "paddle/phi/core/meta_tensor.h"
           "paddle/phi/core/tensor_meta.h"
           "paddle/phi/core/attribute.h"
           "paddle/phi/core/device_context.h"
           "paddle/phi/core/kernel_utils.h"
           "paddle/phi/core/kernel_registry.h"
           "paddle/phi/core/kernel_factory.h"
           "paddle/phi/core/kernel_context.h"
           "paddle/phi/core/infermeta_utils.h"
           "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
           "paddle/fluid/prim/api/composite_backward/composite_double_backward_api.h"
           "paddle/fluid/prim/api/manual_prim/prim_manual_api.h"
           "paddle/fluid/prim/api/api.yaml"
           "python/paddle/incubate/autograd/composite_rules.py"
           "python/paddle/incubate/autograd/primitives.py"
           "python/paddle/autograd/ir_backward.py"
           "python/paddle/autograd/backward_utils.py"
           "paddle/scripts/paddle_build.sh"
           "paddle/phi/infermeta/spmd_rules"
           )

approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
git_files=`git diff --numstat upstream/$BRANCH| wc -l`
git_count=`git diff --numstat upstream/$BRANCH| awk '{sum+=$1}END{print sum}'`
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

function run_tools_test() {
    CUR_PWD=$(pwd)
    cd ${PADDLE_ROOT}/tools
    python $1
    cd ${CUR_PWD}
}

changed_env_var_count=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/paddle | grep 'DEFINE_EXPORTED' | grep -v '@@' | wc -l`
if [[ $changed_env_var_count -gt 0 ]]; then
    echo_line="You must have one RD (lanxianghit (Recommend), phlrain or luotao1 or Aurelius84) approval for changing the FLAGS, which manages the environment variables.\n"
    check_approval 1 lanxianghit phlrain luotao1 Aurelius84
fi

if [[ $git_files -gt 19 || $git_count -gt 999 ]];then
    echo_line="You must have raindrops2sea or XiaoguangHu01 approval for change 20+ files or add than 1000+ lines of content.\n"
    check_approval 1 raindrops2sea XiaoguangHu01
fi

for API_FILE in ${API_FILES[*]}; do
  API_CHANGE=`git diff --name-only upstream/$BRANCH | grep -F "${API_FILE}" | grep -v "/CMakeLists.txt" || true`
  if [ "${API_CHANGE}" ] && [ "${GIT_PR_ID}" != "" ]; then
      # NOTE: per_page=10000 should be ok for all cases, a PR review > 10000 is not human readable.
      # You can use http://caius.github.io/github_id/ to find Github user id.
      # approval_user_list: XiaoguangHu01 46782768,Xreki 12538138,luotao1 6836917,qingqing01 7845005,guoshengCS 14105589,heavengate 12605721,kuke 3064195,Superjomn 328693,lanxianghit 47554610,cyj1986 39645414,hutuxian 11195205,frankwhzhang 20274488,nepeplwu 45024560,raindrops2sea 38231817,chenwhql 22561442,zhiqiu 6888866,seiriosPlus 5442383,gongweibao 10721757,saxon-zh 2870059, zhwesky2010 52485244, Aurelius84 9301846, liym27 33742067, zhhsplendid 7913861, kolinwei 22165420, liuwei1031 46661762, dingjiaweiww 23093488, juncaipeng 52520497, zhangting2020 26615455, Shixiaowei02 39303645, Heeenrrry 28379894,XieYunshen 32428676, Dong Daxiang 35550832, phlrain 43953930, qili93 16605440.
      if [ "${API_FILE}" == "CMakeLists.txt" ];then
          echo_line="You must have one RD (wanghuancoder, luotao1, Aurelius84, XiaoguangHu01 or qili93) approval for CMakeLists.txt, which manages the compilation parameter.\n"
          check_approval 1 wanghuancoder luotao1 Aurelius84 XiaoguangHu01 qili93
      elif [ "${API_FILE}" == "python/paddle/base/__init__.py" ];then
          echo_line="You must have one RD (lanxianghit (Recommend), phlrain, luotao1, Aurelius84 or qili93) approval for the python/paddle/base/init.py, which manages the environment variables.\n"
          check_approval 1 lanxianghit phlrain luotao1 Aurelius84 qili93
      elif [ "${API_FILE}" == "python/requirements.txt" ];then
          echo_line="You must have one RD (phlrain) and one TPM (dingjiaweiww) and one QA (kolinwei) approval for python/requirements.txt, which manages the third-party python package.\n"
          check_approval 3 phlrain dingjiaweiww kolinwei
      elif [ "${API_FILE}" == "paddle/fluid/operators/distributed/send_recv.proto.in" ];then
          echo_line="You must have one RD (gongweibao or seiriosPlus) approval for the paddle/fluid/operators/distributed/send_recv.proto.in, which manages the environment variables.\n"
          check_approval 1 gongweibao seiriosPlus
      elif [ "${API_FILE}" == "paddle/fluid/framework/unused_var_check.cc" ];then
          echo_line="You must have one RD (zhiqiu (Recommend) or chenwhql) approval for the changes of paddle/fluid/framework/unused_var_check.cc, which manages the allow list of operators that have unused input variables. Before change the allow list, please read the specification [https://github.com/PaddlePaddle/Paddle/wiki/OP-Should-Not-Have-Unused-Input] and try to refine code first. \n"
          check_approval 1 zhiqiu chenwhql
      elif [ "${API_FILE}" == "python/paddle/base/tests/unittests/white_list/check_shape_white_list.py" ];then
          echo_line="It is an Op accuracy problem, please take care of it. You must have one RD (hong19860320 (Recommend), luotao1, Aurelisu84, phlrain) approval for the changes of check_shape_white_list.py, which manages the white list of operators with limited input size. Inputs size of all cases in the op test must be greater than or equal to 100. For more information, please refer to: https://github.com/PaddlePaddle/Paddle/wiki/OP-Test-Input-Shape-Requirements. \n"
          check_approval 1 hong19860320 luotao1 Aurelisu84 phlrain
      elif [ "${API_FILE}" == "python/paddle/base/tests/unittests/white_list/op_accuracy_white_list.py" ];then
          echo_line="It is an Op accuracy problem, please take care of it. You must have one RD (juncaipeng (Recommend), zhangting2020 (Recommend) or luotao1 or Aurelius84) approval for the python/paddle/base/tests/unittests/white_list/op_accuracy_white_list.py, which manages the white list of upgrading the precision of op test to float64. For more information, please refer to: https://github.com/PaddlePaddle/Paddle/wiki/Upgrade-OP-Precision-to-Float64. \n"
          check_approval 1 juncaipeng zhangting2020 luotao1 Aurelius84
      elif [ "${API_FILE}" == "python/paddle/base/tests/unittests/white_list/compile_vs_runtime_white_list.py" ];then
           echo_line="You must have one RD (DannyIsFunny (Recommend), luotao1, Aurelius84, phlrain) approval for the python/paddle/base/tests/unittests/white_list/compile_vs_runtime_white_list.py, which manages the white list of compile&runtime lod-level check. For more information, please refer to: https://github.com/PaddlePaddle/Paddle/wiki/Compile_vs_Runtime-Check-Specification. \n"
          check_approval 1 DannyIsFunny luotao1 Aurelius84 phlrain
      elif [ "${API_FILE}" == "python/paddle/base/tests/unittests/white_list/no_check_set_white_list.py" ];then
          echo_line="You must have one RD (cryoco (Recommend), luotao1, Aurelius84 or phlrain) approval for the python/paddle/base/tests/unittests/white_list/no_check_set_white_list.py, which manages the white list of setting no_check_set of check_output. \n"
          check_approval 1 cryoco luotao1 Aurelius84 phlrain
      elif [ "${API_FILE}" == "python/paddle/base/tests/unittests/white_list/check_op_sequence_instance_0_input_white_list.py" ]; then
          echo_line="You must have one RD (luotao1, Aurelisu84, lanxianghit, phlrain) approval for the ${API_FILE}, which manages the white list of instance size 0 input for sequence op test. For more information, please refer to [https://github.com/PaddlePaddle/Paddle/wiki/It-is-required-to-include-LoDTensor-input-with-instance_size=0-in-sequence-OP-test]. \n"
          check_approval 1 luotao1 Aurelisu84 lanxianghit phlrain
      elif [ "${API_FILE}" == "python/paddle/base/tests/unittests/white_list/op_threshold_white_list.py" ];then
          echo_line="It is an Op accuracy problem, please take care of it. You must have one RD (juncaipeng (Recommend), zhangting2020 or luotao1, Aurelius84) approval for the python/paddle/base/tests/unittests/white_list/op_threshold_white_list.py, which manages the white list of error threshold for op test with float64 precision. For more information, please refer to: https://github.com/PaddlePaddle/Paddle/wiki/Upgrade-OP-Precision-to-Float64. \n"
          check_approval 1 juncaipeng zhangting2020 luotao1 Aurelius84
      elif [ "${API_FILE}" == "python/paddle/base/tests/unittests/white_list/check_op_sequence_batch_1_input_white_list.py" ];then
          echo_line="You must have one RD (luotao1, Aurelius84, lanxianghit or phlrain) approval for ${API_FILE}, which manages the white list of batch size 1 input for sequence op test. For more information, please refer to [https://github.com/PaddlePaddle/Paddle/wiki/It-is-required-to-include-LoDTensor-input-with-batch_size=1-in-sequence-OP-test]. \n"
          check_approval 1 luotao1 Aurelius84 lanxianghit phlrain
      elif [ "${API_FILE}" == "python/paddle/base/tests/unittests/white_list/no_grad_set_white_list.py" ];then
          echo_line="You must have one RD (Shixiaowei02 (Recommend), luotao1, Aurelius84 or phlrain) approval for the python/paddle/base/tests/unittests/white_list/no_grad_set_white_list.py, which manages the white list of no_grad_set without value in operators. For more information, please refer to[https://github.com/PaddlePaddle/Paddle/wiki/It's-recommend-to-set-no_grad_set-to-be-None].\n"
          check_approval 1 Shixiaowei02 luotao1 Aurelius84 phlrain
      elif [ "${API_FILE}" == "tools/sampcd_processor.py" ];then
          echo_line="test_sampcd_processor.py will be executed for changed sampcd_processor.py.\n"
          run_tools_test test_sampcd_processor.py
      elif [ "${API_FILE}" == "tools/print_signatures.py" ];then
          echo_line="test_print_signatures.py will be executed for changed print_signatures.py.\n"
          run_tools_test test_print_signatures.py
      elif [ "${API_FILE}" == "tools/checkout_pr_approval.py" ];then
          echo_line="test_checkout_pr_approval.py will be executed for changed checkout_pr_approval.py.\n"
          run_tools_test test_checkout_pr_approval.py
      elif [ "${API_FILE}" == "tools/checkout_api_compatible.py" ];then
          echo_line="test_checkout_api_compatible.py will be executed for changed checkout_api_compatible.py.\n"
          run_tools_test test_checkout_api_compatible.py
      elif [ "${API_FILE}" == "python/paddle/distributed/fleet/__init__.py" ]; then
          echo_line="You must have (fuyinno4 (Recommend), raindrops2sea) approval for ${API_FILE} changes.\n"
          check_approval 1 fuyinno4 raindrops2sea
      elif [ "${API_FILE}" == "python/paddle/distributed/__init__.py" ]; then
          echo_line="You must have (fuyinno4 (Recommend), raindrops2sea) approval for ${API_FILE} changes.\n"
          check_approval 1 fuyinno4 raindrops2sea
      elif [ "${API_FILE}" == "python/paddle/distributed/fleet/launch.py" ]; then
          echo_line="You must have (fuyinno4 (Recommend), raindrops2sea) approval for ${API_FILE} changes.\n"
          check_approval 1 fuyinno4 raindrops2sea
      elif [ "${API_FILE}" == "paddle/scripts/paddle_build.bat" ] || [ "${API_FILE}" == "tools/windows/run_unittests.sh" ]; then
          echo_line="You must have one RD (zhwesky2010 (Recommend), wanghuancoder, luotao1 or Aurelius84) approval for ${API_FILE} changes, which manages the Paddle CI task on Windows.\n"
          check_approval 1 zhwesky2010 wanghuancoder luotao1 Aurelius84
      elif [ "${API_FILE}" == "tools/parallel_UT_rule.py" ]; then
          echo_line="You must have one RD (zhwesky2010 (Recommend), wanghuancoder, luotao1 or Aurelius84) approval for ${API_FILE} changes, which manages the rule of running unittest with a same GPU. If the unittest failed due to Insufficient GPU memory or CUBLAS_STATUS_ALLOC_FAILED, you can remove it from ${API_FILE}.\n"
          check_approval 1 zhwesky2010 wanghuancoder luotao1 Aurelius84
      elif [ "${API_FILE}" == "python/paddle/base/parallel_executor.py" ]; then
          echo_line="You must have one RD (Xreki, luotao1, zhhsplendid or Aurelius84) approval for ${API_FILE}, which manages the underlying code for PaddlePaddle.\n"
          check_approval 1 Xreki luotao1 zhhsplendid Aurelius84
      elif [ "${API_FILE}" == "python/paddle/base/dygraph/layers.py" ] || [ "${API_FILE}" == "paddle/fluid/eager/grad_node_info.h" ] || [ "${API_FILE}" == "paddle/fluid/eager/grad_node_info.cc" ] || [ "${API_FILE}" == "paddle/fluid/eager/grad_tensor_holder.h" ] || [ "${API_FILE}" == "paddle/fluid/eager/grad_tensor_holder.cc" ] || [ "${API_FILE}" == "paddle/fluid/eager/tensor_wrapper.h" ] || [ "${API_FILE}" == "paddle/fluid/eager/autograd_meta.cc"] || [ "${API_FILE}" == "paddle/fluid/eager/autograd_meta.h"] || [ "${API_FILE}" == "paddle/fluid/eager/backward.cc"] || [ "${API_FILE}" == "paddle/fluid/eager/backward.h"]; then
          echo_line="You must have one RD (JiabinYang,chenwhql,phlrain) approval for ${API_FILE}, which manages the underlying code for PaddlePaddle.\n"
          check_approval JiabinYang chenwhql phlrain
      elif [ "${API_FILE}" == "paddle/phi/api/include/tensor.h" ] || [ "${API_FILE}" == "paddle/phi/core/tensor_base.h" ] || [ "${API_FILE}" == "paddle/phi/core/dense_tensor.h" ] || [ "${API_FILE}" == "paddle/phi/core/meta_tensor.h" ] || [ "${API_FILE}" == "paddle/phi/core/tensor_meta.h" ] || [ "${API_FILE}" == "paddle/phi/core/attribute.h" ] || [ "${API_FILE}" == "paddle/phi/core/device_context.h" ] || [ "${API_FILE}" == "paddle/phi/core/kernel_utils.h" ] || [ "${API_FILE}" == "paddle/phi/core/kernel_registry.h" ] || [ "${API_FILE}" == "paddle/phi/core/kernel_factory.h" ] || [ "${API_FILE}" == "paddle/phi/core/kernel_context.h" ] || [ "${API_FILE}" == "paddle/phi/core/infermeta_utils.h" ]; then
            echo_line="You must have one RD (chenwhql, phlrain, zyfncg, YuanRisheng) approval for changing ${API_FILE} , which manages the underlying code for PaddlePaddle PHI Library.\n"
            check_approval chenwhql phlrain zyfncg YuanRisheng
      elif [ "${API_FILE}" == "paddle/fluid/prim/api/composite_backward/composite_backward_api.h" ] || [ "${API_FILE}" == "paddle/fluid/prim/api/manual_prim/prim_manual_api.h" ] || [ "${API_FILE}" == "paddle/fluid/prim/api/api.yaml" ] || [ "${API_FILE}" == "paddle/fluid/prim/api/composite_backward/composite_double_backward_api.h" ]; then
            echo_line="You must have one RD (cxxly(chenxiaoxu), xiaoguoguo626807(wangruting), Charles-hit(wanghao), cyber-pioneer(chenzhuo), JiabinYang) approval for changing ${API_FILE} , which manages the code for PaddlePaddle Composite Bacward Prim API.\n"
            check_approval 1 cxxly xiaoguoguo626807 Charles-hit cyber-pioneer JiabinYang
      elif [ "${API_FILE}" == "python/paddle/incubate/autograd/primitives.py" ] || [ "${API_FILE}" == "python/paddle/incubate/autograd/composite_rules.py" ]; then
            echo_line="You must have one RD (cyber-pioneer(chenzhuo), xiaoguoguo626807(wangruting), Charles-hit(wanghao), JiabinYang) approval for changing ${API_FILE} , which manages the composite rules.\n"
            check_approval 1 cyber-pioneer xiaoguoguo626807 Charles-hit JiabinYang
      elif [ "${API_FILE}" == "python/paddle/autograd/ir_backward.py" ] || [ "${API_FILE}" == "python/paddle/autograd/backward_utils.py" ]; then
            echo_line="You must be approved by Aurelius84(zhangliujie) or cxxly(chenxiaoxu) or xiaoguoguo626807(wangruting) or changeyoung98(chenzhiyang) for python/paddle/autograd/ir_backward.py or python/paddle/autograd/backward_utils.py changes.\n"
            check_approval 1 Aurelius84 cxxly xiaoguoguo626807 changeyoung98
      elif [ "${API_FILE}" == "paddle/scripts/paddle_build.sh" ]; then
	      echo_line="You must have one RD (tianshuo78520a or risemeup1 or zhangbo9674 or XieYunshen) for ${API_FILE} changes, which manages the Paddle CI on Linux.\n "
            check_approval 1 tianshuo78520a risemeup1 zhangbo9674 XieYunshen
      elif [ "${API_FILE}" == "paddle/phi/infermeta/spmd_rules" ]; then
	      echo_line="You must have one RD (liuzhenhai(liuzhenhai93) or liyurui(LiYuRio) or shenliang03(ForFishes) or zhangyichen03(pkuzyc) or chenqiuliang(zhiqiu)) approval for changing ${API_FILE} , which manages the code for spmd_rules.\n"
            check_approval 1 liuzhenhai93 LiYuRio ForFishes pkuzyc zhiqiu
      else
          echo_line="You must have one RD (XiaoguangHu01,chenwhql,zhiqiu,Xreki,luotao1,qili93,Aurelius84) approval for ${API_FILE}, which manages the underlying code for fluid.\n"
          check_approval 1 XiaoguangHu01 chenwhql zhiqiu Xreki luotao1 qili93 Aurelius84
      fi
  fi
done

DEPS_PHI_IN_IR=`git diff --name-only upstream/$BRANCH | grep -E "paddle/pir/" | grep "CMakeList" |xargs -r git diff -U0 upstream/$BRANCH --| grep "^\+" | grep "phi" || true`
echo "DEPS_PHI_IN_IR:${DEPS_PHI_IN_IR}"
if [ "${DEPS_PHI_IN_IR}" ] && [ "${DEPS_PHI_IN_IR}" != "" ]; then
    echo_line="You must have one RD (Aurelius84, phlrain, zhangbo9674, winter-wang) approval for the CMakeLists.txt with DEPS phi* in paddle/pir directory.\n"
    check_approval 1 Aurelius84 phlrain zhangbo9674 winter-wang
fi
FILTER=`git diff --name-only upstream/develop | grep -v "tools/"`
HAS_CONST_CAST=`git diff -U0 upstream/$BRANCH $FILTER | grep '^\+' | grep -o -m 1 "const_cast" || true`
if [ ${HAS_CONST_CAST} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (XiaoguangHu01,chenwhql,zhiqiu,Xreki,luotao1,qili93,Aurelius84) approval for the usage of const_cast.\n"
    check_approval 1 XiaoguangHu01 chenwhql zhiqiu Xreki luotao1 qili93 Aurelius84
fi

HAS_PADDLE_GET=`git diff -U0 upstream/$BRANCH $FILTER |grep "^+" |grep -o -m 1 "paddle::get" || true`
if [ ${HAS_PADDLE_GET} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="paddle::get is not recommended for direct use, because it may throw an bad_variant_access exception without any stack information, so please use PADDLE_GET(_**)(dtype, value) series macros here. If these macros cannot meet your needs, please use try-catch to handle paddle::get and request chenwhql (Recommend), luotao1 or lanxianghit or Aurelius84 review and approve.\n"
    check_approval 1 chenwhql luotao1 lanxianghit Aurelius84
fi

FILTER=`git diff --name-only upstream/develop | grep -v "tools/"`
HAS_LEGACY_KERNEL_REGISTRATION=`git diff -U0 upstream/$BRANCH $FILTER | grep '^\+' | grep -oE -m 1 "REGISTER_OP[A-Z_]{1,9}KERNEL[_FUNCTOR|_WITH_CUSTOM_TYPE|_EX]*" || true`
if [ ${HAS_LEGACY_KERNEL_REGISTRATION} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="In principle, adding an OpKernel needs to be in the phi/kernels directory. If you must add an OpKernel in the fluid/operators directory, please request one of the RD (chenwhql, zyfncg, YuanRisheng, phlrain) review and approve.\n"
    check_approval 1 chenwhql zyfncg YuanRisheng phlrain
fi

DIFF_OUTPUT=$(git diff --unified=0 upstream/$BRANCH)
# check if any .cc or .cu file in the phi/kernels/ directory is changed and if any template is added
if echo "$DIFF_OUTPUT" | grep -q 'diff --git a/paddle/phi/kernels/.*\.cc b/paddle/phi/kernels/.*\.cc\|diff --git a/paddle/phi/kernels/.*\.cu b/paddle/phi/kernels/.*\.cu'; then
    if echo "$DIFF_OUTPUT" | grep -q '+.*template <'; then
        echo "A C++ template is added in .cc or .cu file in the phi/kernels directory,which can lead to an overly large size of the compiled .o file, resulting in a failure in multi-architecture compilation!"
        echo_line="You must have one RD (risemeup1 or Galaxy1458) approval for the change of C++ template.\n"
        check_approval 1 risemeup1 Galaxy1458
    fi
fi

PYTHON_FILE_ADDED_LINES=$(git diff -U0 upstream/$BRANCH -- 'python/*.py' |grep "^+")
IF_USE_SUBPROCESS=`echo $PYTHON_FILE_ADDED_LINES | grep -B5 --no-group-separator "subprocess\." || true`
if [[ ${IF_USE_SUBPROCESS} ]]; then
    echo_line="You must have one RD (wanghuancoder(Recommend), Aurelius84, 2742195759, SigureMo) approval for using subprocess, which may cause security problem.\n"
    check_approval 1 wanghuancoder Aurelius84 2742195759 SigureMo
fi
IF_USE_EVAL=`echo $PYTHON_FILE_ADDED_LINES | grep -B5 --no-group-separator "[^\w\d_]eval([^()]*[a-zA-Z0-9_])" || true`
if [[ ${IF_USE_EVAL} ]]; then
    echo_line="You must have one RD (wanghuancoder(Recommend), Aurelius84, 2742195759, SigureMo) approval for using eval, which may cause security problem.\n"
    check_approval 1 wanghuancoder Aurelius84 2742195759 SigureMo
fi

HAS_DEFINE_FLAG=`git diff -U0 upstream/$BRANCH |grep -o -m 1 "DEFINE_int32" |grep -o -m 1 "DEFINE_bool" | grep -o -m 1 "DEFINE_string" || true`
if [ ${HAS_DEFINE_FLAG} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD lanxianghit approval for the usage (either add or delete) of DEFINE_int32/DEFINE_bool/DEFINE_string flag.\n"
    check_approval 1 lanxianghit
fi

NO_NPU_FILE=`git diff --name-only upstream/$BRANCH | grep -v "_npu.py"`
HAS_UNITTEST_SKIP=`git diff -U0 upstream/$BRANCH ${NO_NPU_FILE} | grep "^+[[:space:]]\{0,\}@unittest.skip" || true`
if [ "${HAS_UNITTEST_SKIP}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="Unittest is not allowed to be disabled.\nYou must have one RD (kolinwei(Recommend), wanghuancoder, luotao1, QingshuChen, qili93 or ZzSean or Aurelius84) approval for the usage of @unittest.skip or @unittest.skipIf.\n${HAS_UNITTEST_SKIP}\n"
    check_approval 1 kolinwei wanghuancoder luotao1 QingshuChen qili93 ZzSean Aurelius84
  fi

HAS_MODIFIED_DEMO_CMAKE=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/inference/api/demo_ci/CMakeLists.txt" || true`
if [ "${HAS_MODIFIED_DEMO_CMAKE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (Superjomn (Recommend), Shixiaowei02, luotao1 or Aurelius84) approval for paddle/fluid/inference/api/demo_ci/CMakeLists.txt.\nwhich manages the compilation parameter of inference demo\n"
    check_approval 1 Superjomn Shixiaowei02 luotao1 Aurelius84
  fi

HAS_MODIFIED_DECLARATIONS=`git diff -U0 upstream/$BRANCH |grep "^+" |grep "paddle/phi/kernels/declarations.h" || true`
if [ "${HAS_MODIFIED_DECLARATIONS}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must be approved by chenwhql or zyfncg for paddle/phi/kernels/declarations.h using. Thanks!\n"
    check_approval 1 chenwhql zyfncg
  fi

HAS_USED_CCTESTOLD=`git diff -U0 upstream/$BRANCH |grep "cc_test_old" || true`
if [ "${HAS_USED_CCTESTOLD}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must be approved by phlrain or risemeup1 or zhangbo9674 or Galaxy1458 for using cc_test_old. Thanks!\n"
    check_approval 1 phlrain risemeup1 zhangbo9674 Galaxy1458
fi

HAS_USED_CCTEST=`git diff -U0 upstream/$BRANCH |grep "cc_test" || true`
if [ "${HAS_USED_CCTEST}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="Paddle utest will gradually discard cc_test\n  instead, the paddle_test is recommended,\n if you must use cc_test, you must be approved by risemeup1 or zhangbo9674 or Galaxy1458 for using cc_test. Thanks!\n"
    check_approval 1 risemeup1 zhangbo9674 Galaxy1458
fi

HAS_CREATE_NEW_PASS=`git diff -U0 upstream/$BRANCH |grep "paddle/pir/include/pass/pass.h" || true`
if [ "${HAS_CREATE_NEW_PASS}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="\n If you implement a new Pass, you must be approved by yuanlehome or zyfncg. Thanks!\n"
    check_approval 1 yuanlehome zyfncg
fi

HAS_MODIFIED_API_COMPAT_YAML=`git diff --name-only upstream/$BRANCH | grep "paddle/phi/api/yaml/op_compat.yaml" || true`
if [ "${HAS_MODIFIED_API_COMPAT_YAML}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must be approved by chenwhql or zyfncg or heavyrain-lzy for paddle/phi/api/yaml/op_compat.yaml changes, which manages the extra params of Op and name mapping between Yaml and OpMaker. In order to ensure compatibility of framework, this file isn't allowed to be modified at will!\n"
    check_approval 1 chenwhql zyfncg heavyrain-lzy
fi

HAS_MODIFIED_API_FW_BW_YAML=`git diff --name-only upstream/$BRANCH | grep -E "paddle/phi/api/yaml/ops.yaml|paddle/phi/api/yaml/backward.yaml" || true`
if [ "${HAS_MODIFIED_API_FW_BW_YAML}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must be approved by chenwhql or zyfncg or heavyrain-lzy for paddle/phi/api/yaml/ops.yaml or paddle/phi/api/yaml/backward.yaml changes, which manage the generated code for the C++ OP. You can only change them according to the specification at the begining of this two file.\n"
    check_approval 1 chenwhql zyfncg heavyrain-lzy
fi

HAS_MODIFIED_FRAMEWORK_EXECUTOR=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/framework/new_executor" || true`
if [ "${HAS_MODIFIED_FRAMEWORK_EXECUTOR}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (From00, zhangbo9674) approval for file changes in paddle/fluid/framework/new_executor.\n"
    check_approval 1 From00 zhangbo9674
fi


HAS_MODIFIED_DRR_INCLUDE_DIR=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/pir/drr/include" || true`
if [ "${HAS_MODIFIED_DRR_INCLUDE_DIR}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (yuanlehome, zyfncg) approval for file changes in paddle/fluid/pir/drr/include.\n"
    check_approval 1 yuanlehome zyfncg
fi


HAS_MODIFIED_PIR_INCLUDE_DIR=`git diff --name-only upstream/$BRANCH | grep "paddle/pir/include" || true`
if [ "${HAS_MODIFIED_PIR_INCLUDE_DIR}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (yuanlehome, winter-wang, zhangbo9674) approval for file changes in paddle/pir/include.\n"
    check_approval 1 yuanlehome winter-wang zhangbo9674 
fi

HAS_MODIFIED_API_GENE=`git diff --name-only upstream/$BRANCH | grep "paddle/phi/api/yaml/generator" || true`
if [ "${HAS_MODIFIED_API_GENE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (zyfncg, chenwhql, YuanRisheng, phlrain, heavyrain-lzy) approval for file changes in paddle/phi/api/yaml/generator, which manages the generated code for C++ API in paddle/phi/api/lib/api.cc.\n"
    check_approval 1 zyfncg chenwhql YuanRisheng phlrain heavyrain-lzy
fi

HAS_MODIFIED_EAGER_GENE=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/eager/auto_code_generator" || true`
if [ "${HAS_MODIFIED_EAGER_GENE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (JiabinYang, chenwhql, zyfncg, phlrain, heavyrain-lzy) approval for file changes in paddle/fluid/eager/auto_code_generator, which manages the generated code for dygraph functions in paddle/fluid/eager/api/generated.\n"
    check_approval 1 JiabinYang chenwhql zyfncg phlrain heavyrain-lzy
fi

HAS_MODIFIED_OPERATOR_GENE=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/operators/generator" || true`
if [ "${HAS_MODIFIED_OPERATOR_GENE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (zyfncg, chenwhql, iclementine, phlrain, heavyrain-lzy) approval for file changes in paddle/fluid/operators/generator, which manages the generated code for OpMaker in paddle/fluid/operators/(generated_op.cc | sparse_generated_op.cc)\n"
    check_approval 1 zyfncg chenwhql iclementine phlrain heavyrain-lzy
fi

HAS_MODIFIED_SETUP_IN=`git diff --name-only upstream/$BRANCH | grep "python/setup.py.in" || true`
if [ "${HAS_MODIFIED_SETUP_IN}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (chenwhql, zyfncg, YuanRisheng, phlrain) approval for file changes in python/setup.py.in, which manages the header files that can be used from outside of framework.\n"
    check_approval 1 chenwhql zyfncg YuanRisheng phlrain
fi

HAS_MODIFIED_SETUP=`git diff --name-only upstream/$BRANCH | grep "${PADDLE_ROOT}/setup.py" || true`
if [ "${HAS_MODIFIED_SETUP}" != "" ] || ([ "${HAS_MODIFIED_SETUP_IN}" != "" ] && [ "${HAS_MODIFIED_SETUP}" == "" ]); then
    echo_line="You must have one RD (risemeup1, zhangbo9674) approval for file changes in setup.py or setup.py and python/setup.py.in are not changed synchronously.\n"
    check_approval 1 risemeup1 zhangbo9674
fi

HAS_MODIFIED_STATIC_BUILD=`git diff --name-only upstream/$BRANCH | grep "new_executor/interpreter/static_build.cc" || true`
if [ "${HAS_MODIFIED_STATIC_BUILD}" != "" ] && [ "${GIT_PR_ID}" != ""]; then
    echo_line="You must have one RD (From00 or zhiqiu) approval for file changes in new_executor/interpreter/static_build.cc.\n"
    check_approval 1 From00 zhiqiu
fi

HAS_MODIFIED_ENFORCE_SYNTAX=`git diff --name-only upstream/$BRANCH | grep -E "IR_ENFORCE|CHECK_EQ|CHECK_NE|CHECK_LT|CHECK_LE|CHECK_GE|CHECK_GT|LOG\(FATAL\)" || true`
if [ "${HAS_MODIFIED_ENFORCE_SYNTAX}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (rismeup1 or winter-wang) approval for using 'IR_ENFORCE, COMMON_ENFORCE, CHECK_EQ, CHECK_NE, CHECK_LT, CHECK_LE, CHECK_GE, CHECK_GT, LOG(FATAL)', it is recommended to use paddle_ENFORCE as a replacement.\n"
    check_approval 1 risemeup1 winter-wang
fi


HAS_MODIFIED_TARGET_FOR_AUTO_PARALLEL_CI=`git diff --name-only upstream/$BRANCH | grep "tools/auto_parallel/target_path_lists.sh" || true`
if [ "${HAS_MODIFIED_TARGET_FOR_AUTO_PARALLEL_CI}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (zhiqiu(Recommend) or chenwhql) approval for file changes in tools/auto_parallel/target_path_lists.sh.\n"
    check_approval 1 zhiqiu chenwhql
fi

HAS_MODIFIED_PY_FLUID=`git diff --name-only upstream/$BRANCH | grep "python/paddle/fluid" || true`
if [ "${HAS_MODIFIED_PY_FLUID}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (zoooo0820(Recommend), or jeff41404) approval for file changes in python/paddle/fluid, because fluid API has been removed.\n"
    check_approval 1 zoooo0820 jeff41404
fi

ALL_PADDLE_ENFORCE=`git diff -U0 upstream/$BRANCH |grep "^+" |grep -zoE "PADDLE_ENFORCE\(.[^,\);]+.[^;]*\);\s" || true`
if [ "${ALL_PADDLE_ENFORCE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="PADDLE_ENFORCE is not recommended. Please use PADDLE_ENFORCE_EQ/NE/GT/GE/LT/LE or PADDLE_ENFORCE_NOT_NULL or PADDLE_ENFORCE_GPU_SUCCESS instead, see [ https://github.com/PaddlePaddle/Paddle/wiki/PADDLE_ENFORCE-Rewriting-Specification ] for details.\nYou must have one RD (chenwhql (Recommend), luotao1 (Recommend) or lanxianghit or Aurelius84) approval for the usage (either add or delete) of PADDLE_ENFORCE.\n${ALL_PADDLE_ENFORCE}\n"
    check_approval 1 chenwhql luotao1 lanxianghit Aurelius84
fi

ALL_ADDED_LINES=$(git diff -U0 upstream/$BRANCH |grep "^+" || true)
ALL_PADDLE_CHECK=$(echo $ALL_ADDED_LINES |grep -zoE "(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\(.[^,\);]*.[^;]*\);\s" || true)
VALID_PADDLE_CHECK=$(echo "$ALL_PADDLE_CHECK" | grep -zoE '(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\(([^,;]+,)*[^";]*errors::.[^"]*".[^";]{20,}.[^;]*\);\s' || true)
INVALID_PADDLE_CHECK=$(echo "$ALL_PADDLE_CHECK" |grep -vxF "$VALID_PADDLE_CHECK" || true)
if [ "${INVALID_PADDLE_CHECK}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="The error message you wrote in PADDLE_ENFORCE{_**} or PADDLE_THROW does not meet our error message writing specification. Possible errors include 1. the error message is empty / 2. the error message is too short / 3. the error type is not specified. Please read the specification [ https://github.com/PaddlePaddle/Paddle/wiki/Paddle-Error-Message-Writing-Specification ], then refine the error message. If it is a mismatch, please request chenwhql (Recommend), luotao1 or lanxianghit or Aurelius84 review and approve.\nThe PADDLE_ENFORCE{_**} or PADDLE_THROW entries that do not meet the specification are as follows:\n${INVALID_PADDLE_CHECK}\n"
    check_approval 1 chenwhql luotao1 lanxianghit Aurelius84
fi

EMPTY_GRAD_OP_REGISTERED=`echo $ALL_ADDED_LINES |grep -zoE "REGISTER_OP_WITHOUT_GRADIENT\([^;.]*\)[;\s]" || echo $ALL_ADDED_LINES |grep -zoE "[[:graph:]]*EmptyGradOpMaker<[[:graph:]]*>" || true`
if [ "${EMPTY_GRAD_OP_REGISTERED}" != "" ] && [ "${GIT_PT_ID}" != "" ]; then
    echo_line="You must have one RD (phlrain, XiaoguangHu01, kolinwei or JiabinYang) approval for the usage of REGISTER_OP_WITHOUT_GRADIENT or EmptyGradOpMaker.\nThe code that do not meet the specification are as follows:\n${EMPTY_GRAD_OP_REGISTERED}\n"
    check_approval 1 phlrain XiaoguangHu01 kolinwei JiabinYang
fi

INVALID_UNITTEST_ASSERT_CHECK=`echo "$ALL_ADDED_LINES" | grep -zoE '\+\s+((assert\s+)|(self\.assert(True|Equal)\())(\s*\+\s*)?(np|numpy)\.(allclose|array_equal)[^+]*' || true`
if [ "${INVALID_UNITTEST_ASSERT_CHECK}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="It is recommended to use 'np.testing.assert_allclose' and 'np.testing.array_equal' instead of 'self.assertTrue(np.allclose(...))' and 'self.assertTrue(np.array_equal(...))'.\nPlease modify the code below. If anything is unclear, please read the specification [ https://github.com/PaddlePaddle/community/blob/master/rfcs/CodeStyle/20220805_code_style_improvement_for_unittest.md#background ]. If it is a mismatch, please request qili93 (Recommend) or luotao1 or Aurelius84 review and approve.\nThe code that do not meet the specification are as follows:\n${INVALID_UNITTEST_ASSERT_CHECK}\n"
    check_approval 1 qili93 luotao1 Aurelius84
fi

DEPRECATED_FLAKE8=`git diff --name-only upstream/$BRANCH | grep ".flake8" || true`
if [ "${DEPRECATED_FLAKE8}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one SigureMo or gouzil approval for file changes in .flake8, we are planned to replace Flake8 with Ruff in the future.\n"
    check_approval 1 SigureMo gouzil
fi

TEST_FILE_ADDED_LINES=$(git diff -U0 upstream/$BRANCH -- test |grep "^+")
ENABLE_TO_STATIC_CHECK=`echo "$TEST_FILE_ADDED_LINES" | grep "enable_to_static(" || true`
if [ "${ENABLE_TO_STATIC_CHECK}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (SigureMo, Aurelius84 or 2742195759) approval for using 'paddle.jit.enable_to_static', we recommend using 'enable_to_static_guard' in the related test files.\n"
    check_approval 1 SigureMo Aurelius84 2742195759
fi

HAS_MODIFIED_DY2ST_TEST_FILES=$(git diff --name-only upstream/$BRANCH | grep "test/dygraph_to_static/test_" || true)
if [ "${HAS_MODIFIED_DY2ST_TEST_FILES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    error_lines=`python ${PADDLE_ROOT}/test/dygraph_to_static/check_approval.py ${HAS_MODIFIED_DY2ST_TEST_FILES}`
    if [ $? -ne 0 ]; then
        echo_line="Your PR does not meet Dy2St unittest dev guide, please check https://github.com/PaddlePaddle/Paddle/issues/61464 for details.\n"
        echo_line=${echo_line}"Errors are as follows:\n"
        echo_line=${echo_line}${error_lines}"\n"
        echo_line=${echo_line}"You can run following command to fix the errors:\n"
        echo_line=${echo_line}"    python test/dygraph_to_static/check_approval.py "$(echo ${HAS_MODIFIED_DY2ST_TEST_FILES} | tr "\n" " ")"\n"
        echo_line=${echo_line}"If you believe this is a false positive, please request one of the RD (SigureMo, Aurelius84, 2742195759 or gouzil) approval for the changes.\n"
        check_approval 1 SigureMo Aurelius84 2742195759 gouzil
    fi
fi

HAS_MODIFIED_DY2ST_TEST_TENSOR_ATTR_CONSISTENCY=$(git diff --name-only upstream/$BRANCH | grep "test/dygraph_to_static/test_tensor_attr_consistency.py" || true)
if [ "${HAS_MODIFIED_DY2ST_TEST_TENSOR_ATTR_CONSISTENCY}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (SigureMo, Aurelius84, 2742195759 or gouzil) approval for file changes in test/dygraph_to_static/test_tensor_attr_consistency.py.\n"
    check_approval 1 SigureMo Aurelius84 2742195759 gouzil
fi

HAS_MODIFIED_PHI_FILES=`git diff --name-only upstream/$BRANCH | grep "paddle/phi/" || true`
PHI_INCLUDE_FLUID_FILES=""
for CHANGE_FILE in ${HAS_MODIFIED_PHI_FILES}; do
    PHI_DIR_ADDED_LINES=`git diff -U0 upstream/$BRANCH -- ${PADDLE_ROOT}/${CHANGE_FILE} | grep "^+" | grep "#include \"paddle/fluid/" || true`
    if [ "${PHI_DIR_ADDED_LINES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
        PHI_INCLUDE_FLUID_FILES="${PHI_INCLUDE_FLUID_FILES} ${CHANGE_FILE}"
    fi
done
if [ "${PHI_INCLUDE_FLUID_FILES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (chenwhql, YuanRisheng or zyfncg) approval for the including paddle/fluid header in paddle/phi files(${PHI_INCLUDE_FLUID_FILES}).\n"
    check_approval 1 chenwhql YuanRisheng zyfncg
fi

HAS_MODIFIED_PHI_HEADER_FILES=`git diff --name-only upstream/$BRANCH | grep "paddle/phi/.*\.h" || true`
PHI_INCLUDE_THIRD_PARTY_FILES=""
for CHANGE_FILE in ${HAS_MODIFIED_PHI_HEADER_FILES}; do
    PHI_DIR_ADDED_LINES=`git diff -U0 upstream/$BRANCH -- ${PADDLE_ROOT}/${CHANGE_FILE} | grep "^+" | grep -E "#include \"gflags/gflags.h\"|#include \"glog/logging.h\"" || true`
    if [ "${PHI_DIR_ADDED_LINES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
        PHI_INCLUDE_THIRD_PARTY_FILES="${PHI_INCLUDE_THIRD_PARTY_FILES} ${CHANGE_FILE}"
    fi
done
if [ "${PHI_INCLUDE_THIRD_PARTY_FILES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (jiahy0825, zyfncg, chenwhql, YuanRisheng or heavyrain-lzy) approval for including \"gflags/gflags.h\" or \"glog/logging.h\" headerfile in paddle/phi headerfiles(${PHI_INCLUDE_THIRD_PARTY_FILES}). Recommend including third party headers in phi source files(*.cc) instead of phi headerfiles(*.h). Because if phi headerfiles include third party headers like \"gflags.h\" or \"logging.h\", error might occur when outside developers use phi headerfiles directly.\n"
    check_approval 1 jiahy0825 zyfncg chenwhql YuanRisheng heavyrain-lzy
fi

HAS_MODIFIED_PADDLE_API_FILES=`git diff --name-only upstream/$BRANCH | grep "paddle/.*\.h" || true`
INCLUDE_PADDLE_API_FILES=""
for CHANGE_FILE in ${HAS_MODIFIED_PHI_HEADER_FILES}; do
    PADDLE_API_ADDED_LINES=`git diff -U0 upstream/$BRANCH -- ${PADDLE_ROOT}/${CHANGE_FILE} | grep -w "PADDLE_API" || true`
    if [ "${PADDLE_API_ADDED_LINES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
        INCLUDE_PADDLE_API_FILES="${INCLUDE_PADDLE_API_FILES} ${CHANGE_FILE}"
    fi
done
if [ "${INCLUDE_PADDLE_API_FILES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (jiahy0825, zyfncg, chenwhql) or PM (sunzhongkai588, Ligoml) approval for code changes about PADDLE_API. If you add a new PADDLE_API, please make sure you have written detailed comments about the parameter and usage of this PADDLE_API .\n"
    check_approval 1 jiahy0825 zyfncg chenwhql sunzhongkai588 Ligoml
fi

HAS_MODIFIED_PHI_OR_FLUID_FILES=`git diff --name-only upstream/$BRANCH | grep -E "paddle/phi|paddle/fluid" || true`
USE_MUTABLE_DATA_FILES=""
for CHANGE_FILE in ${HAS_MODIFIED_PHI_OR_FLUID_FILES}; do
    ADDED_LINES=`git diff -U0 upstream/$BRANCH -- ${PADDLE_ROOT}/${CHANGE_FILE} | grep "^+" | grep -w "mutable_data" || true`
    if [ "${ADDED_LINES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
        USE_MUTABLE_DATA_FILES="${USE_MUTABLE_DATA_FILES} ${CHANGE_FILE}"
    fi
done
if [ "${USE_MUTABLE_DATA_FILES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You can not use the DenseTensor::mutable_data() method in files(${USE_MUTABLE_DATA_FILES}). If you want to alloc memory, use phi::DeviceContext::Alloc() or phi::DeviceContext::HostAlloc() instead and if you want to get mutable data, use DenseTensor::data(). If you have any questions, you can have one RD (chenwhql, Shixiaowei02, YuanRisheng, zyfncg or From00) review and approve.\n"
    check_approval 1 chenwhql Shixiaowei02 YuanRisheng zyfncg From00
fi

ALL_CHANGE_FILES=`git diff --numstat upstream/$BRANCH | awk '{print $3}' | grep ".py"`
ALL_OPTEST_BAN_DYGRAPH_MESSAGE=""
for CHANGE_FILE in ${ALL_CHANGE_FILES}; do
    ALL_OPTEST_BAN_DYGRAPH=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${CHANGE_FILE} | grep "+" | grep "check_dygraph=" || true`
    if [ "${ALL_OPTEST_BAN_DYGRAPH}" != "" ]; then
        ALL_OPTEST_BAN_DYGRAPH_MESSAGE="${ALL_OPTEST_BAN_DYGRAPH_MESSAGE} ${CHANGE_FILE} : \n${ALL_OPTEST_BAN_DYGRAPH} \n"
    fi
done
if [ "${ALL_OPTEST_BAN_DYGRAPH_MESSAGE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
  echo_line="Developers are not allowed to set the check_dygraph field directly, which is set to True by default. If you need to change the check_dygraph field, you must have one RD (phlrain (Recommend), fuyinno4, QingshuChen (Recommend for kunlun) or lanxianghit) review and approve. \nThe code that do not meet the specification are as follows:\n${ALL_OPTEST_BAN_DYGRAPH_MESSAGE}\n"
    check_approval 1 phlrain fuyinno4 QingshuChen lanxianghit
fi

ALL_CHANGE_YAML_FILES=`git diff --numstat upstream/$BRANCH | awk '{print $3}' | grep ".yaml"`
BAN_COMP_MESSAGE=""
for CHANGE_FILE in ${ALL_CHANGE_YAML_FILES}; do
    ALL_ITEM_BAN_COMP=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${CHANGE_FILE} | grep "composite" || true`
    if [ "${ALL_ITEM_BAN_COMP}" != "" ]; then
        BAN_COMP_MESSAGE="${BAN_COMP_MESSAGE} ${CHANGE_FILE} : \n${ALL_ITEM_BAN_COMP} \n"
    fi
done
if [ "${BAN_COMP_MESSAGE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
  echo_line="If you need to change the key composite, you must have one RD (Charles-hit(wanghao), cyber-pioneer(chenzhuo), cxxly(chenxiaoxu)) review and approve. \nThe code that do not meet the specification are as follows:\n${BAN_COMP_MESSAGE}\n"
    check_approval 1 Charles-hit cyber-pioneer cxxly
fi

NEW_OP_ADDED=`git diff --name-only --diff-filter=A upstream/$BRANCH |grep -oE ".+_op..*" || true`
if [ "${NEW_OP_ADDED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    GET_KERNEL_TYPE_FUNC_CNT=`git diff -U0 --diff-filter=A upstream/$BRANCH |grep "+" |grep -czoE "GetExpectedKernelType[(][^(){}]+[)][^{]+[{][^}]+[}]" || true`
    INDICATE_VAR_DTYPE_CNT=`git diff -U0 --diff-filter=A upstream/$BRANCH |grep "+" |grep -co "IndicateVarDataType" || true`
    if [ ${GET_KERNEL_TYPE_FUNC_CNT} -gt ${INDICATE_VAR_DTYPE_CNT} ]; then
        echo_line="If you override GetExpectedKernelType method of OperatorWithKernel, please use OperatorWithKernel::IndicateVarDataType() method to get specific input variable's dtype, which checked whether the input variable is initialized (The details in https://github.com/PaddlePaddle/FluidDoc/pull/1527). If you don't use this method to check, you must have one RD (chenwhql (Recommend) , luotao1 or lanxianghit or Aurelius84) approval for the usage of other methods.\n"
        check_approval 1 chenwhql luotao1 lanxianghit Aurelius84
    fi
fi

HAS_OPERATORBASE_FLAG=`git diff -U0 --diff-filter=A upstream/$BRANCH | grep -E "public[[:space:]]+.*OperatorBase" || true`
if [ "${HAS_OPERATORBASE_FLAG}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="In order to support dynamic graph, all ops are not recommended to inherit OperatorBase. Please use OperatorWithKernel instead.\nYou must have one RD (phlrain (Recommend), luotao1, lanxianghit, XiaoguangHu01, or qili93 or Aurelius84) approval for the inherit of OperatorBase.\nYou inherit the OperatorBase class. The corresponding lines are as follows:\n${HAS_OPERATORBASE_FLAG}"
    check_approval 1 phlrain luotao1 lanxianghit XiaoguangHu01 qili93 Aurelius84
fi

HAS_INPLACE_TESTS=`git diff -U0 upstream/$BRANCH |grep "+" |grep -E "inplace_atol[[:space:]]*=.*" || true`
if [ "${HAS_INPLACE_TESTS}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="The calculation results of setting inplace enabled and disabled must be equal, that is, it's not recommended to set inplace_atol.\n If you do need to use inplace_atol, you must have one RD (XiaoguangHu01, lanxianghit, phlrain, luotao1, QingshuChen, Aurelius84) approval for the usage of inplace_atol.\nThe corresponding lines are as follows:\n${HAS_INPLACE_TESTS}\n"
    check_approval 1 XiaoguangHu01 lanxianghit phlrain luotao1 QingshuChen Aurelius84
fi

OP_FILE_CHANGED=`git diff --name-only --diff-filter=AMR upstream/$BRANCH |grep -oE ".+_op..*" || true`
if [ "${OP_FILE_CHANGED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    ERROR_LINES=""
    for OP_FILE in ${OP_FILE_CHANGED};
    do
        CHECK_OBJECT_FLAGS=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${OP_FILE} |grep "+" |grep -E "ShareDataWith[(]|ShareBufferWith[(]" || true`
        if [ "${CHECK_OBJECT_FLAGS}" != "" ]; then
            ERROR_LINES="${ERROR_LINES}\n${OP_FILE}${CHECK_OBJECT_FLAGS}\n"
        fi
    done
    if [ "${ERROR_LINES}" != "" ]; then
        ERROR_LINES=${ERROR_LINES//+/'\n+\t'}
        echo_line="Using ShareDataWith or ShareBufferWith is not recommended. You must have one RD's (zhhsplendid (Recommend), zhiqiu or luotao1 or lanxianghit or qili93 or Aurelius84) approval to use these methods. For more information, please refer to https://github.com/PaddlePaddle/Paddle/wiki/ShareDataWith-is-prohibited-in-OP. The error lines are as follows:${ERROR_LINES}"
        check_approval 1 zhhsplendid zhiqiu luotao1 lanxianghit qili93 Aurelius84
    fi
fi

CMAKE_FILE_CHANGED=`git diff --name-only --diff-filter=AMR upstream/$BRANCH |grep -E "\.cmake|CMakeLists\.txt"  || true`
if [ "${CMAKE_FILE_CHANGED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    ERROR_LINES=""
    for CMAKE_FILE in ${CMAKE_FILE_CHANGED};
    do
        CHECK_OBJECT_FLAGS=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${CMAKE_FILE} |grep "+" |grep -E "\-Wno\-error" || true`
        if [ "${CHECK_OBJECT_FLAGS}" != "" ]; then
            ERROR_LINES="${ERROR_LINES}\n${CMAKE_FILE}${CHECK_OBJECT_FLAGS}\n"
        fi
    done
    if [ "${ERROR_LINES}" != "" ]; then
        ERROR_LINES=${ERROR_LINES//+/'\n+\t'}
        echo_line="Change compilation flag of warnings is not recommended. You must have one RD's (zhiqiu (Recommend), luotao1 or phlrain or Aurelius84) approval to use these methods. "
        check_approval 1 zhiqiu luotao1 phlrain Aurelius84
    fi
fi

NEW_OP_TEST_ADDED=`git diff --name-only --diff-filter=AMR upstream/$BRANCH |grep -oE "test_.*.\.py" || true`
if [ "${NEW_OP_TEST_ADDED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    CHECK_OUTPUT=`git diff -U5 --diff-filter=AMR upstream/$BRANCH |grep "self\.check_output(a*t*o*l*=*[0-9]"|grep "+" || true`
    CHECK_OUTPUT_WITH_PLACE=`git diff -U5 --diff-filter=AMR upstream/$BRANCH |grep -A2 "self\.check_output_with_place" |grep ", [atol*,0-9]"|grep "+" || true`
    CHECK_GRAD=`git diff -U5 --diff-filter=AMR upstream/$BRANCH |grep -A5 -E "self\.check_grad|self\.check_grad_with_place"|grep "max_relative_error=" |grep "+" || true`
    CHECK_GRAD_CHECK=`git diff -U5 --diff-filter=AMR upstream/$BRANCH |grep -A2 -E "checker\.double_grad_check"|grep "eps=|atol=|rtol=" |grep "+" || true`
    CHECK_WHOLE=$CHECK_OUTPUT$CHECK_OUTPUT_WITH_PLACE$CHECK_GRAD$CHECK_GRAD_CHECK
    if [ "${CHECK_WHOLE}" != "" ] ; then
        CHECK_OP=${CHECK_WHOLE//+/'\n+'}
        echo_line="Please use the default precision parameters of 'atol, rtol, eps, max_relative_error'. If you don't use the default value, you must have one RD (Xreki (Recommend), fuyinno4, QingshuChen(Recommend for kunlun), zhiqiu or qili93 (Recommend for NPU) , luotao1, lanxianghit, phlrain or ZzSean or Aurelius84) approval for the usage of other values. The detailed information is in the link: https://github.cor/PaddlePaddle/Paddle/wiki/OP-test-accuracy-requirements. The error line is ${CHECK_OP}\n"
        check_approval 1 Xreki fuyinno4 QingshuChen zhiqiu qili93 luotao1 lanxianghit phlrain ZzSean Aurelius84
    fi
fi

UNITTEST_FILE_CHANGED=`git diff --name-only --diff-filter=AM upstream/$BRANCH |grep -E "test_.*.\.py" || true`
if [ "${UNITTEST_FILE_CHANGED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    ERROR_LINES=""
    for TEST_FILE in ${UNITTEST_FILE_CHANGED};
    do
        HAS_SKIP_CHECK_GRAD_CI=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${TEST_FILE} |grep "@skip_check_grad_ci" || true`
        if [ "${HAS_SKIP_CHECK_GRAD_CI}" != "" ]; then
            ERROR_LINES="${ERROR_LINES}\n${TEST_FILE}\n${HAS_SKIP_CHECK_GRAD_CI}\n"
        fi
    done
    if [ "${ERROR_LINES}" != "" ]; then
        ERROR_LINES=${ERROR_LINES//+/'\n+\t'}
        echo_line="It is an Op accuracy problem, please take care of it. You must have one RD (zhangting2020 (Recommend), luotao1 or phlrain, qili93, QingshuChen or Aurelius84) approval for the usage (either add or delete) of @skip_check_grad_ci. For more information, please refer to: https://github.com/PaddlePaddle/Paddle/wiki/Gradient-Check-Is-Required-for-Op-Test. The corresponding lines are as follows:\n${ERROR_LINES}\n"
        check_approval 1 zhangting2020 luotao1 phlrain qili93 QingshuChen Aurelius84
    fi
fi

RUNTYPE_FILE_CHANGED=`git diff --name-only --diff-filter=AM upstream/$BRANCH|grep -E "CMakeLists.txt"||true`
if [ "${RUNTYPE_FILE_CHANGED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    for CMAKELISTS_FILE in ${RUNTYPE_FILE_CHANGED};
    do
        RUNTYPE_ADD=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${CMAKELISTS_FILE} |grep "^+" |grep -E "SERIAL|RUN_TYPE=EXCLUSIVE|RUN_TYPE=DIST|RUN_TYPE=HYBRID|RUN_TYPE=NIGHTLY|RUN_TYPE=EXCLUSIVE:NIGHTLY|RUN_TYPE=DIST:NIGHTLY|PROPERTIES[[:space:]]+TIMEOUT" || true`
    if [[ ${RUNTYPE_ADD} != "" ]];then
        RUNTYPE_ADD_LINES="${RUNTYPE_ADD_LINES}\n${CMAKELISTS_FILE}\n${RUNTYPE_ADD}\n"
    fi
    done
    if [[ ${RUNTYPE_ADD_LINES} != "" ]];then
        echo_line="You must have one QA (XieYunshen(Recommend) or chalsliu) approval for setting parameter RUN_TYPE as EXCLUSIVE, DIST, HYBRID, NIGHTLY, EXCLUSIVE:NIGHTLY or DISTNIGHTLY, or setting parameter SERIAL, or setting TIMEOUT properties.\nThe corresponding lines are as follows:\n${RUNTYPE_ADD_LINES}\nFor more information, please refer to:https://github.com/PaddlePaddle/Paddle/wiki/PaddlePaddle-Unit-test-specification"
    check_approval 1 XieYunshen chalsliu
    fi
fi

SKIP_CI=`git log --pretty=oneline|grep $AGILE_REVISION |grep -w "test=document_fix" || true`
if [[ ${SKIP_CI} ]];then
    echo_line="You must have one RD (tianshuo78520a (Recommend), zhiqiu, phlrain ) or PM (Ligoml) approval you add test=document_fix method in commit skips CI"
    check_approval 1 tianshuo78520a zhiqiu phlrain Ligoml
fi

# Get the list of PR authors with unresolved unit test issues
pip install PyGithub
# For getting PR related data
wget https://sys-p0.bj.bcebos.com/blk/block.txt --no-check-certificate --no-proxy
wget https://sys-p0.bj.bcebos.com/bk-ci/bk.txt --no-check-certificate --no-proxy
HASUTFIXED=`python ${PADDLE_ROOT}/tools/check_ut.py | grep "has unit-test to be fixed" || true`
if [ "${HASUTFIXED}" != "" ]; then
  echo_line="${HASUTFIXED} You must have one RD (chalsliu (Recommend) or kolinwei) approval.\n"
  check_approval 1 chalsliu kolinwei
fi

HASUTFIXED=`python ${PADDLE_ROOT}/tools/check_ut.py | grep "has benchmark issue to be fixed" || true`
if [ "${HASUTFIXED}" != "" ]; then
    echo_line="${HASUTFIXED} You must have one RD (hysunflower or xiegegege or Xreki) approval.\n"
  check_approval 1 hysunflower xiegegege Xreki
fi

# NOTE(Avin0323): Files with the name "unity_build_rule.cmake" are rules used
# by Unity Build to combine source files. Changes to these rules may cause
# errors in the compilation. Specific personal are required to approve the
# modification of these files.
UNITYBUILD_RULE_CHANGED=$(git diff --name-only upstream/$BRANCH |
                          grep "unity_build_rule.cmake" || true)
if [ -n "${UNITYBUILD_RULE_CHANGED}" -a -n "${GIT_PR_ID}" ]; then
    echo_line="You must have one RD (Avin0323(Recommend) or zhwesky2010 or
               wanghuancoder or luotao1 or Aurelius84) approval for modifying
               unity_build_rule.cmake which the rules of Unity Build."
    echo_line=$(echo ${echo_line})
    # Avin0323(23427135) zhwesky2010(52485244)
    # wanghuancoder(26922892) luotao1(6836917)
    check_approval 1 Avin0323 zhwesky2010 wanghuancoder luotao1 Aurelius84
fi

if [ -n "${echo_list}" ];then
  echo "****************"
  echo -e "${echo_list[@]}"
  echo "There are ${failed_num} approved errors."
  echo "****************"
fi

if [ -n "${echo_list}" ]; then
  exit 6
fi

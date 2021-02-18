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
           "python/requirements.txt"
           "python/paddle/fluid/__init__.py"
           "python/paddle/fluid/compiler.py"
           "python/paddle/fluid/parallel_executor.py"
           "python/paddle/fluid/framework.py"
           "python/paddle/fluid/backward.py"
           "paddle/fluid/operators/distributed/send_recv.proto.in"
           "paddle/fluid/framework/unused_var_check.cc"
           "paddle/fluid/pybind/op_function_generator.cc"
           "python/paddle/fluid/tests/unittests/white_list/check_shape_white_list.py"
           "python/paddle/fluid/tests/unittests/white_list/op_accuracy_white_list.py"
           "python/paddle/fluid/tests/unittests/white_list/compile_vs_runtime_white_list.py"
           "python/paddle/fluid/tests/unittests/white_list/no_check_set_white_list.py"
           "python/paddle/fluid/tests/unittests/white_list/check_op_sequence_instance_0_input_white_list.py"
           "python/paddle/fluid/tests/unittests/white_list/op_threshold_white_list.py"
           "python/paddle/fluid/tests/unittests/white_list/check_op_sequence_batch_1_input_white_list.py"
           "python/paddle/fluid/tests/unittests/white_list/no_grad_set_white_list.py"
           "tools/wlist.json"
           "paddle/scripts/paddle_build.bat"
           "tools/windows/run_unittests.sh"
           "tools/parallel_UT_rule.py"
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


if [[ $git_files -gt 19 || $git_count -gt 999 ]];then
    echo_line="You must have Dianhai approval for change 20+ files or add than 1000+ lines of content.\n"
    check_approval 1 38231817
fi

for API_FILE in ${API_FILES[*]}; do
  API_CHANGE=`git diff --name-only upstream/$BRANCH | grep "${API_FILE}" | grep -v "/CMakeLists.txt" || true`
  if [ "${API_CHANGE}" ] && [ "${GIT_PR_ID}" != "" ]; then
      # NOTE: per_page=10000 should be ok for all cases, a PR review > 10000 is not human readable.
      # You can use http://caius.github.io/github_id/ to find Github user id.
      # approval_user_list: XiaoguangHu01 46782768,Xreki 12538138,luotao1 6836917,qingqing01 7845005,guoshengCS 14105589,heavengate 12605721,kuke 3064195,Superjomn 328693,lanxianghit 47554610,cyj1986 39645414,hutuxian 11195205,frankwhzhang 20274488,nepeplwu 45024560,Dianhai 38231817,chenwhql 22561442,zhiqiu 6888866,seiriosPlus 5442383,gongweibao 10721757,saxon-zh 2870059, zhouwei25 52485244, Aurelius84 9301846, liym27 33742067, zhhsplendid 7913861, kolinwei 22165420, liuwei1031 46661762, swtkiwi 27208573, juncaipeng 52520497, zhangting2020 26615455, Shixiaowei02 39303645, Heeenrrry 28379894,XieYunshen 32428676, Dong Daxiang 35550832, phlrain 43953930.
      if [ "${API_FILE}" == "CMakeLists.txt" ];then
          echo_line="You must have one RD (luotao1 or XiaoguangHu01) approval for CMakeLists.txt, which manages the compilation parameter.\n"
          check_approval 1 6836917 46782768
      elif [ "${API_FILE}" == "python/paddle/fluid/__init__.py" ];then
          echo_line="You must have one RD (lanxianghit (Recommend) or luotao1) approval for the python/paddle/fluid/init.py, which manages the environment variables.\n"
          check_approval 1 6836917 47554610
      elif [ "${API_FILE}" == "python/requirements.txt" ];then
          echo_line="You must have one RD (phlrain) and one TPM (swtkiwi) and one QA (kolinwei) approval for python/requirements.txt, which manages the third-party python package.\n"
          check_approval 3 43953930 27208573 22165420
      elif [ "${API_FILE}" == "paddle/fluid/operators/distributed/send_recv.proto.in" ];then
          echo_line="You must have one RD (gongweibao or seiriosPlus) approval for the paddle/fluid/operators/distributed/send_recv.proto.in, which manages the environment variables.\n"
          check_approval 1 10721757 5442383
      elif [ "${API_FILE}" == "paddle/fluid/framework/unused_var_check.cc" ];then
          echo_line="You must have one RD (zhiqiu (Recommend) or luotao1) approval for the changes of paddle/fluid/framework/unused_var_check.cc, which manages the allow list of operators that have unused input variables. Before change the allow list, please read the specification [https://github.com/PaddlePaddle/Paddle/wiki/OP-Should-Not-Have-Unused-Input] and try to refine code first. \n"
          check_approval 1 6888866 6836917
      elif [ "${API_FILE}" == "paddle/fluid/pybind/op_function_generator.cc" ];then
          echo_line="You must have one RD (zhiqiu (Recommend) , phlrain) approval for the changes of paddle/fluid/pybind/op_function_generator.cc, which manages the logic of automatic generating op functions for dygraph. \n"
          check_approval 1 6888866 43953930
      elif [ "${API_FILE}" == "python/paddle/fluid/tests/unittests/white_list/check_shape_white_list.py" ];then
          echo_line="It is an Op accuracy problem, please take care of it. You must have one RD (hong19860320 (Recommend), luotao1, phlrain) approval for the changes of check_shape_white_list.py, which manages the white list of operators with limited input size. Inputs size of all cases in the op test must be greater than or equal to 100. For more information, please refer to: https://github.com/PaddlePaddle/Paddle/wiki/OP-Test-Input-Shape-Requirements. \n"
          check_approval 1 9973393 6836917 43953930
      elif [ "${API_FILE}" == "python/paddle/fluid/tests/unittests/white_list/op_accuracy_white_list.py" ];then
          echo_line="It is an Op accuracy problem, please take care of it. You must have one RD (juncaipeng (Recommend), zhangting2020 (Recommend) or luotao1) approval for the python/paddle/fluid/tests/unittests/white_list/op_accuracy_white_list.py, which manages the white list of upgrading the precision of op test to float64. For more information, please refer to: https://github.com/PaddlePaddle/Paddle/wiki/Upgrade-OP-Precision-to-Float64. \n"
          check_approval 1 52520497 26615455 6836917
      elif [ "${API_FILE}" == "python/paddle/fluid/tests/unittests/white_list/compile_vs_runtime_white_list.py" ];then
           echo_line="You must have one RD (DannyIsFunny (Recommend), luotao1, phlrain) approval for the python/paddle/fluid/tests/unittests/white_list/compile_vs_runtime_white_list.py, which manages the white list of compile&runtime lod-level check. For more information, please refer to: https://github.com/PaddlePaddle/Paddle/wiki/Compile_vs_Runtime-Check-Specification. \n"
          check_approval 1 45189361 6836917 43953930
      elif [ "${API_FILE}" == "python/paddle/fluid/tests/unittests/white_list/no_check_set_white_list.py" ];then
          echo_line="You must have one RD (cryoco (Recommend), luotao1 or phlrain) approval for the python/paddle/fluid/tests/unittests/white_list/no_check_set_white_list.py, which manages the white list of setting no_check_set of check_output. \n"
          check_approval 1 12407750 6836917 43953930
      elif [ "${API_FILE}" == "python/paddle/fluid/tests/unittests/white_list/check_op_sequence_instance_0_input_white_list.py" ]; then
          echo_line="You must have one RD (luotao1, phlrain) approval for the ${API_FILE}, which manages the white list of instance size 0 input for sequence op test. For more information, please refer to [https://github.com/PaddlePaddle/Paddle/wiki/It-is-required-to-include-LoDTensor-input-with-instance_size=0-in-sequence-OP-test]. \n"
          check_approval 1 6836917 43953930
      elif [ "${API_FILE}" == "python/paddle/fluid/tests/unittests/white_list/op_threshold_white_list.py" ];then
          echo_line="It is an Op accuracy problem, please take care of it. You must have one RD (juncaipeng (Recommend), zhangting2020 or luotao1) approval for the python/paddle/fluid/tests/unittests/white_list/op_threshold_white_list.py, which manages the white list of error threshold for op test with float64 precision. For more information, please refer to: https://github.com/PaddlePaddle/Paddle/wiki/Upgrade-OP-Precision-to-Float64. \n"
          check_approval 1 52520497 26615455 6836917
      elif [ "${API_FILE}" == "python/paddle/fluid/tests/unittests/white_list/check_op_sequence_batch_1_input_white_list.py" ];then
          echo_line="You must have one RD (luotao1 or phlrain) approval for ${API_FILE}, which manages the white list of batch size 1 input for sequence op test. For more information, please refer to [https://github.com/PaddlePaddle/Paddle/wiki/It-is-required-to-include-LoDTensor-input-with-batch_size=1-in-sequence-OP-test]. \n"
          check_approval 1 6836917 43953930
      elif [ "${API_FILE}" == "python/paddle/fluid/tests/unittests/white_list/no_grad_set_white_list.py" ];then
          echo_line="You must have one RD (Shixiaowei02 (Recommend), luotao1 or phlrain) approval for the python/paddle/fluid/tests/unittests/white_list/no_grad_set_white_list.py, which manages the white list of no_grad_set without value in operators. For more information, please refer to[https://github.com/PaddlePaddle/Paddle/wiki/It's-recommend-to-set-no_grad_set-to-be-None].\n"
          check_approval 1 39303645 6836917 43953930
      elif [ "${API_FILE}" == "tools/wlist.json" ];then
          echo_line="You must have one TPM (jzhang533) approval for the api whitelist for the tools/wlist.json.\n"
          check_approval 1 29231
      elif [ "${API_FILE}" == "python/paddle/distributed/fleet/__init__.py" ]; then
	      echo_line="You must have (guru4elephant,raindrops2sea) approval for ${API_FILE} changes "
	      check_approval 1 35550832 38231817
      elif [ "${API_FILE}" == "python/paddle/distributed/__init__.py" ]; then
	      echo_line="You must have (guru4elephant,raindrops2sea) approval for ${API_FILE} changes "
	      check_approval 1 35550832 38231817
      elif [ "${API_FILE}" == "paddle/scripts/paddle_build.bat" ] || [ "${API_FILE}" == "tools/windows/run_unittests.sh" ]; then
	      echo_line="You must have one RD (zhouwei25 (Recommend), luotao1) approval for ${API_FILE} changes, which manages the Paddle CI task on Windows.\n"
	      check_approval 1 52485244 6836917
      elif [ "${API_FILE}" == "tools/parallel_UT_rule.py" ]; then
	      echo_line="You must have one RD (zhouwei25 (Recommend), luotao1) approval for ${API_FILE} changes, which manages the rule of running unittest with a same GPU. If the unittest failed due to Insufficient GPU memory or CUBLAS_STATUS_ALLOC_FAILED, you can remove it from ${API_FILE}.\n"
	      check_approval 1 52485244 6836917
      elif [ "${API_FILE}" == "python/paddle/fluid/parallel_executor.py" ]; then
          echo_line="You must have one RD (Xreki,luotao1,zhhsplendid) approval for ${API_FILE}, which manages the underlying code for PaddlePaddle.\n"
          check_approval 1 12538138 6836917 7913861
      else
          echo_line="You must have one RD (XiaoguangHu01,Xreki,luotao1) approval for ${API_FILE}, which manages the underlying code for fluid.\n"
          check_approval 1 46782768 12538138 6836917
      fi
  fi
done

FILTER=`git diff --name-only upstream/develop | grep -v "tools/"`
HAS_CONST_CAST=`git diff -U0 upstream/$BRANCH $FILTER | grep '^\+' | grep -o -m 1 "const_cast" || true`
if [ ${HAS_CONST_CAST} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (XiaoguangHu01,Xreki,luotao1) approval for the usage of const_cast.\n"
    check_approval 1 46782768 12538138 6836917
fi

HAS_BOOST_GET=`git diff -U0 upstream/$BRANCH $FILTER |grep "^+" |grep -o -m 1 "boost::get" || true`
if [ ${HAS_BOOST_GET} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="boost::get is not recommended, because it may throw an bad_get exception without any stack information, so please use BOOST_GET(_**)(dtype, value) series macros here. If these macros cannot meet your needs, please use try-catch to handle boost::get and request chenwhql (Recommend), luotao1 or lanxianghit review and approve.\n"
    check_approval 1 6836917 47554610 22561442
fi

HAS_LOG_FATAL=`git diff -U0 upstream/$BRANCH $FILTER |grep "^+" |grep -o -m 1 "LOG(FATAL)" || true`
if [ ${HAS_LOG_FATAL} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="LOG(FATAL) is not recommended, because it will throw exception without standard stack information, so please use PADDLE_THROW macro here. If you have to use LOG(FATAL) here, please request chenwhql (Recommend), luotao1 or lanxianghit review and approve.\n"
    check_approval 1 6836917 47554610 22561442
fi

HAS_DEFINE_FLAG=`git diff -U0 upstream/$BRANCH |grep -o -m 1 "DEFINE_int32" |grep -o -m 1 "DEFINE_bool" | grep -o -m 1 "DEFINE_string" || true`
if [ ${HAS_DEFINE_FLAG} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD lanxianghit approval for the usage (either add or delete) of DEFINE_int32/DEFINE_bool/DEFINE_string flag.\n"
    check_approval 1 47554610
fi

HAS_UNITTEST_SKIP=`git diff -U0 upstream/$BRANCH | grep "^+[[:space:]]\{0,\}@unittest.skip" || true`
if [ "${HAS_UNITTEST_SKIP}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="Unittest is not allowed to be disabled.\nYou must have one RD (kolinwei(Recommend), or luotao1) approval for the usage of @unittest.skip or @unittest.skipIf.\n${HAS_UNITTEST_SKIP}\n"
    check_approval 1 22165420 6836917 46661762
  fi

HAS_MODIFIED_DEMO_CMAKE=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/inference/api/demo_ci/CMakeLists.txt" || true`
if [ "${HAS_MODIFIED_DEMO_CMAKE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (Superjomn (Recommend), luotao1) approval for paddle/fluid/inference/api/demo_ci/CMakeLists.txt.\nwhich manages the compilation parameter of inference demo\n"
    check_approval 1 328693 6836917
  fi

ALL_PADDLE_ENFORCE=`git diff -U0 upstream/$BRANCH |grep "^+" |grep -zoE "PADDLE_ENFORCE\(.[^,\);]+.[^;]*\);\s" || true`
if [ "${ALL_PADDLE_ENFORCE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="PADDLE_ENFORCE is not recommended. Please use PADDLE_ENFORCE_EQ/NE/GT/GE/LT/LE or PADDLE_ENFORCE_NOT_NULL or PADDLE_ENFORCE_CUDA_SUCCESS instead, see [ https://github.com/PaddlePaddle/Paddle/wiki/PADDLE_ENFORCE-Rewriting-Specification ] for details.\nYou must have one RD (chenwhql (Recommend) , luotao1 (Recommend) or lanxianghit) approval for the usage (either add or delete) of PADDLE_ENFORCE.\n${ALL_PADDLE_ENFORCE}\n"
    check_approval 1 6836917 47554610 22561442
fi

ALL_ADDED_LINES=`git diff -U0 upstream/$BRANCH |grep "^+" || true`
ALL_PADDLE_CHECK=`echo $ALL_ADDED_LINES |grep -zoE "(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\(.[^,\);]*.[^;]*\);\s" || true`
VALID_PADDLE_CHECK=`echo "$ALL_PADDLE_CHECK" | grep -zoE '(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\((.[^,;]+,)*.[^";]*(errors::).[^"]*".[^";]{20,}.[^;]*\);\s' || true`
INVALID_PADDLE_CHECK=`echo "$ALL_PADDLE_CHECK" |grep -vxF "$VALID_PADDLE_CHECK" || true`
if [ "${INVALID_PADDLE_CHECK}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="The error message you wrote in PADDLE_ENFORCE{_**} or PADDLE_THROW does not meet our error message writing specification. Possible errors include 1. the error message is empty / 2. the error message is too short / 3. the error type is not specified. Please read the specification [ https://github.com/PaddlePaddle/Paddle/wiki/Paddle-Error-Message-Writing-Specification ], then refine the error message. If it is a mismatch, please request chenwhql (Recommend), luotao1 or lanxianghit review and approve.\nThe PADDLE_ENFORCE{_**} or PADDLE_THROW entries that do not meet the specification are as follows:\n${INVALID_PADDLE_CHECK}\n"
    check_approval 1 6836917 47554610 22561442
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
  echo_line="Developers are not allowed to set the check_dygraph field directly, which is set to True by default. If you need to change the check_dygraph field, you must have one RD (phlrain (Recommend), fuyinno4 (Recommend for kunlun) or lanxianghit) review and approve. \nThe code that do not meet the specification are as follows:\n${ALL_OPTEST_BAN_DYGRAPH_MESSAGE}\n"
    check_approval 1 43953930 47554610 35824027
fi

NEW_OP_ADDED=`git diff --name-only --diff-filter=A upstream/$BRANCH |grep -oE ".+_op..*" || true`
if [ "${NEW_OP_ADDED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    GET_KERNEL_TYPE_FUNC_CNT=`git diff -U0 --diff-filter=A upstream/$BRANCH |grep "+" |grep -czoE "GetExpectedKernelType[(][^(){}]+[)][^{]+[{][^}]+[}]" || true`
    INDICATE_VAR_DTYPE_CNT=`git diff -U0 --diff-filter=A upstream/$BRANCH |grep "+" |grep -co "IndicateVarDataType" || true`
    if [ ${GET_KERNEL_TYPE_FUNC_CNT} -gt ${INDICATE_VAR_DTYPE_CNT} ]; then
        echo_line="If you override GetExpectedKernelType method of OperatorWithKernel, please use OperatorWithKernel::IndicateVarDataType() method to get specific input variable's dtype, which checked whether the input variable is initialized (The details in https://github.com/PaddlePaddle/FluidDoc/pull/1527). If you don't use this method to check, you must have one RD (chenwhql (Recommend) , luotao1 or lanxianghit) approval for the usage of other methods.\n"
        check_approval 1 6836917 47554610 22561442
    fi
fi

HAS_OPERATORBASE_FLAG=`git diff -U0 --diff-filter=A upstream/$BRANCH | grep -E "public[[:space:]]+.*OperatorBase" || true`
if [ "${HAS_OPERATORBASE_FLAG}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="In order to support dynamic graph, all ops are not recommended to inherit OperatorBase. Please use OperatorWithKernel instead.\nYou must have one RD (phlrain (Recommend), luotao1, lanxianghit or XiaoguangHu01) approval for the inherit of OperatorBase.\nYou inherit the OperatorBase class. The corresponding lines are as follows:\n${HAS_OPERATORBASE_FLAG}"
    check_approval 1 43953930 6836917 47554610 46782768
fi

HAS_INPLACE_TESTS=`git diff -U0 upstream/$BRANCH |grep "+" |grep -E "inplace_atol[[:space:]]*=.*" || true`
if [ "${HAS_INPLACE_TESTS}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="The calculation results of setting inplace enabled and disabled must be equal, that is, it's not recommended to set inplace_atol.\n If you do need to use inplace_atol, you must have one RD (XiaoguangHu01, lanxianghit, phlrain, luotao1) approval for the usage of inplace_atol.\nThe corresponding lines are as follows:\n${HAS_INPLACE_TESTS}\n"
    check_approval 1 46782768 47554610 43953930 6836917
fi

OP_FILE_CHANGED=`git diff --name-only --diff-filter=AMR upstream/$BRANCH |grep -oE ".+_op..*" || true`
if [ "${OP_FILE_CHANGED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    for OP_FILE in ${OP_FILE_CHANGED};
    do
        CHECK_OBJECT_FLAGS=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${OP_FILE} |grep "+" |grep -E "ShareDataWith[(]|ShareBufferWith[(]" || true`
        if [ "${CHECK_OBJECT_FLAGS}" != "" ]; then
            ERROR_LINES="${ERROR_LINES}\n${OP_FILE}${CHECK_OBJECT_FLAGS}\n"
        fi
    done
    if [ "${ERROR_LINES}" != "" ]; then
        ERROR_LINES=${ERROR_LINES//+/'\n+\t'}
        echo_line="Using ShareDataWith or ShareBufferWith is not recommended. You must have one RD's (zhhsplendid (Recommend), zhiqiu or luotao1 or lanxianghit) approval to use these methods. For more information, please refer to https://github.com/PaddlePaddle/Paddle/wiki/ShareDataWith-is-prohibited-in-OP. The error lines are as follows:${ERROR_LINES}"
        check_approval 1 6836917 6888866 47554610 7913861
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
        echo_line="Please use the default precision parameters of 'atol, rtol, eps, max_relative_error'. If you don't use the default value, you must have one RD (Xreki (Recommend), fuyinno4 (Recommend for kunlun), luotao1, lanxianghit or phlrain) approval for the usage of other values. The detailed information is in the link: https://github.cor/PaddlePaddle/Paddle/wiki/OP-test-accuracy-requirements. The error line is ${CHECK_OP}\n"
        check_approval 1 6836917 47554610 12538138 43953930 35824027
    fi
fi

UNITTEST_FILE_CHANGED=`git diff --name-only --diff-filter=AM upstream/$BRANCH |grep -E "test_.*.\.py" || true`
if [ "${UNITTEST_FILE_CHANGED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    for TEST_FILE in ${UNITTEST_FILE_CHANGED};
    do
        HAS_SKIP_CHECK_GRAD_CI=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${TEST_FILE} |grep "@skip_check_grad_ci" || true`
        if [ "${HAS_SKIP_CHECK_GRAD_CI}" != "" ]; then
            ERROR_LINES="${ERROR_LINES}\n${TEST_FILE}\n${HAS_SKIP_CHECK_GRAD_CI}\n"
        fi
    done
    if [ "${ERROR_LINES}" != "" ]; then
        ERROR_LINES=${ERROR_LINES//+/'\n+\t'}
        echo_line="It is an Op accuracy problem, please take care of it. You must have one RD (zhangting2020 (Recommend), luotao1 or phlrain) approval for the usage (either add or delete) of @skip_check_grad_ci. For more information, please refer to: https://github.com/PaddlePaddle/Paddle/wiki/Gradient-Check-Is-Required-for-Op-Test. The corresponding lines are as follows:\n${ERROR_LINES}\n"
        check_approval 1 26615455 6836917 43953930
    fi
fi

RUNTYPE_FILE_CHANGED=`git diff --name-only --diff-filter=AM upstream/$BRANCH|grep -E "CMakeLists.txt"||true`
if [ "${RUNTYPE_FILE_CHANGED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    for CMAKELISTS_FILE in ${RUNTYPE_FILE_CHANGED};
    do
        RUNTYPE_ADD=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${CMAKELISTS_FILE} |grep "^+" |grep -E "RUN_TYPE=EXCLUSIVE|RUN_TYPE=DIST|RUN_TYPE=NIGHTLY|RUN_TYPE=EXCLUSIVE:NIGHTLY|RUN_TYPE=DIST:NIGHTLY|PROPERTIES[[:space:]]+TIMEOUT" || true`
	if [[ ${RUNTYPE_ADD} != "" ]];then
	    RUNTYPE_ADD_LINES="${RUNTYPE_ADD_LINES}\n${CMAKELISTS_FILE}\n${RUNTYPE_ADD}\n"
	fi
    done
    if [[ ${RUNTYPE_ADD_LINES} != "" ]];then
        echo_line="You must have one QA (XieYunshen(Recommend) or chalsliu) approval for setting parameter RUN_TYPE as EXCLUSIVE, DIST, NIGHTLY, EXCLUSIVE:NIGHTLY or DISTNIGHTLY, or setting TIMEOUT properties.\nThe corresponding lines are as follows:\n${RUNTYPE_ADD_LINES}\nFor more information, please refer to:https://github.com/PaddlePaddle/Paddle/wiki/PaddlePaddle-Unit-test-specification"
	check_approval 1 32428676 45041955
    fi
fi

# Get the list of PR authors with unresolved unit test issues
pip install PyGithub
# For getting PR related data
wget https://sys-p0.bj.bcebos.com/blk/block.txt --no-check-certificate --no-proxy
wget https://sys-p0.bj.bcebos.com/bk-ci/bk.txt --no-check-certificate --no-proxy
HASUTFIXED=`python ${PADDLE_ROOT}/tools/check_ut.py | grep "has unit-test to be fixed" || true`
if [ "${HASUTFIXED}" != "" ]; then
  echo_line="${HASUTFIXED} You must have one RD (chalsliu (Recommend) or kolinwei) approval.\n"
  check_approval 1 45041955 22165420
fi

HASUTFIXED=`python ${PADDLE_ROOT}/tools/check_ut.py | grep "has benchmark issue to be fixed" || true`
if [ "${HASUTFIXED}" != "" ]; then
    echo_line="${HASUTFIXED} You must have one RD (hysunflower or xiegegege or Xreki) approval.\n"
  check_approval 1 52739577 46314656 12538138
fi

# NOTE(Avin0323): Files with the name "unity_build_rule.cmake" are rules used
# by Unity Build to combine source files. Changes to these rules may cause
# errors in the compilation. Specific personal are required to approve the
# modification of these files.
UNITYBUILD_RULE_CHANGED=$(git diff --name-only upstream/$BRANCH |
                          grep "unity_build_rule.cmake" || true)
if [ -n "${UNITYBUILD_RULE_CHANGED}" -a -n "${GIT_PR_ID}" ]; then
    echo_line="You must have one RD (Avin0323(Recommend) or zhouwei25 or
               wanghuancoder or luotao1) approval for modifying
               unity_build_rule.cmake which the rules of Unity Build."
    echo_line=$(echo ${echo_line})
    # Avin0323(16167147) zhouwei25(52485244)
    # wanghuancoder(26922892) luotao1(6836917)
    check_approval 1 16167147 52485244 26922892 6836917
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
